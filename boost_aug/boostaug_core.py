import itertools
import random
import shutil
import time
import os
import numpy as np
import torch
import tqdm
from autocuda import auto_cuda
from pyabsa.core.tad.classic.__bert__ import TADBERT
from pyabsa.core.tad.prediction.tad_classifier import TADTextClassifier
from pyabsa.core.tc.prediction.text_classifier import TextClassifier

from boost_aug import __version__

from findfile import find_cwd_files, find_cwd_dir, find_dir, find_files, find_dirs, find_file
from pyabsa import APCCheckpointManager, APCModelList, TCCheckpointManager, BERTTCModelList, TCDatasetList, TCConfigManager, APCConfigManager, TADConfigManager, TADCheckpointManager
from pyabsa.core.apc.prediction.sentiment_classifier import SentimentClassifier

from termcolor import colored

from pyabsa.functional import Trainer
from pyabsa.functional.config.config_manager import ConfigManager
from pyabsa.functional.dataset import DatasetItem
from pyabsa import ABSADatasetList

from transformers import BertForMaskedLM, DebertaV2ForMaskedLM, AutoConfig, AutoTokenizer, RobertaForMaskedLM
import tempfile
import git


def rename(src, tgt):
    if src == tgt:
        return
    if os.path.exists(tgt):
        remove(tgt)
    os.rename(src, tgt)


def remove(p):
    if os.path.exists(p):
        os.remove(p)


class AugmentBackend:
    EDA = 'EDA'
    ContextualWordEmbsAug = 'ContextualWordEmbsAug'
    RandomWordAug = 'RandomWordAug'
    AntonymAug = 'AntonymAug'
    SplitAug = 'SplitAug'
    BackTranslationAug = 'BackTranslationAug'
    SpellingAug = 'SpellingAug'


class ABSCBoostAug:

    def __init__(self,
                 ROOT: str = '',
                 BOOSTING_FOLD=5,
                 CLASSIFIER_TRAINING_NUM=2,
                 CONFIDENCE_THRESHOLD=0.99,
                 AUGMENT_NUM_PER_CASE=10,
                 WINNER_NUM_PER_CASE=10,
                 PERPLEXITY_THRESHOLD=4,
                 AUGMENT_PCT=0.1,
                 AUGMENT_BACKEND=AugmentBackend.EDA,
                 USE_CONFIDENCE=True,
                 USE_PERPLEXITY=True,
                 USE_LABEL=True,
                 device='cuda'
                 ):
        """

        :param ROOT: The path to save intermediate checkpoint
        :param BOOSTING_FOLD: Number of splits in crossing boosting augment
        :param CLASSIFIER_TRAINING_NUM: Number of pre-trained inference model using for confidence calculation
        :param CONFIDENCE_THRESHOLD: Confidence threshold used for augmentations filtering
        :param AUGMENT_NUM_PER_CASE: Number of augmentations per example
        :param WINNER_NUM_PER_CASE: Number of selected augmentations per example in confidence ranking
        :param PERPLEXITY_THRESHOLD: Perplexity threshold used for augmentations filtering
        :param AUGMENT_PCT: Word change probability used in backend augment method
        :param AUGMENT_BACKEND: Augmentation backend used for augmentations generation, e.g., EDA, ContextualWordEmbsAug
        """

        assert hasattr(AugmentBackend, AUGMENT_BACKEND)
        if not ROOT or not os.path.exists(ROOT):
            self.ROOT = os.getenv('$HOME') if os.getenv('$HOME') else os.getcwd()
        else:
            self.ROOT = ROOT

        self.BOOSTING_FOLD = BOOSTING_FOLD
        self.CLASSIFIER_TRAINING_NUM = CLASSIFIER_TRAINING_NUM
        self.CONFIDENCE_THRESHOLD = CONFIDENCE_THRESHOLD
        self.AUGMENT_NUM_PER_CASE = AUGMENT_NUM_PER_CASE if AUGMENT_NUM_PER_CASE > 0 else 1
        self.WINNER_NUM_PER_CASE = WINNER_NUM_PER_CASE
        self.PERPLEXITY_THRESHOLD = PERPLEXITY_THRESHOLD
        self.AUGMENT_PCT = AUGMENT_PCT
        self.AUGMENT_BACKEND = AUGMENT_BACKEND
        self.USE_CONFIDENCE = USE_CONFIDENCE
        self.USE_PERPLEXITY = USE_PERPLEXITY
        self.USE_LABEL = USE_LABEL
        self.device = device

        if self.AUGMENT_BACKEND in 'EDA':
            # Here are some augmenters from https://github.com/QData/TextAttack
            from textattack.augmentation import EasyDataAugmenter as Aug
            # Alter default values if desired
            self.augmenter = Aug(pct_words_to_swap=self.AUGMENT_PCT, transformations_per_example=self.AUGMENT_NUM_PER_CASE)
        else:
            # Here are some augmenters from https://github.com/makcedward/nlpaug
            import nlpaug.augmenter.word as naw
            if self.AUGMENT_BACKEND in 'ContextualWordEmbsAug':
                self.augmenter = naw.ContextualWordEmbsAug(
                    model_path='roberta-base', action="substitute", aug_p=self.AUGMENT_PCT, device=self.device)
            elif self.AUGMENT_BACKEND in 'RandomWordAug':
                self.augmenter = naw.RandomWordAug(action="swap")
            elif self.AUGMENT_BACKEND in 'AntonymAug':
                self.augmenter = naw.AntonymAug()
            elif self.AUGMENT_BACKEND in 'SplitAug':
                self.augmenter = naw.SplitAug()
            elif self.AUGMENT_BACKEND in 'BackTranslationAug':
                self.augmenter = naw.BackTranslationAug(from_model_name='facebook/wmt19-en-de',
                                                        to_model_name='facebook/wmt19-de-en',
                                                        device=self.device
                                                        )
            elif self.AUGMENT_BACKEND in 'SpellingAug':
                self.augmenter = naw.SpellingAug()

    def get_mlm_and_tokenizer(self, sent_classifier, config):

        if isinstance(sent_classifier, SentimentClassifier):
            base_model = sent_classifier.model.bert.base_model
        else:
            base_model = sent_classifier.bert.base_model
        pretrained_config = AutoConfig.from_pretrained(config.pretrained_bert)
        try:
            if 'deberta-v3' in config.pretrained_bert:
                MLM = DebertaV2ForMaskedLM(pretrained_config).to(sent_classifier.opt.device)
                MLM.deberta = base_model
            elif 'roberta' in config.pretrained_bert:
                MLM = RobertaForMaskedLM(pretrained_config).to(sent_classifier.opt.device)
                MLM.roberta = base_model
            else:
                MLM = BertForMaskedLM(pretrained_config).to(sent_classifier.opt.device)
                MLM.bert = base_model
        except Exception as e:
            self.device = auto_cuda()
            if 'deberta-v3' in config.pretrained_bert:
                MLM = DebertaV2ForMaskedLM(pretrained_config).to(self.device)
                MLM.deberta = base_model
            elif 'roberta' in config.pretrained_bert:
                MLM = RobertaForMaskedLM(pretrained_config).to(self.device)
                MLM.roberta = base_model
            else:
                MLM = BertForMaskedLM(pretrained_config).to(self.device)
                MLM.bert = base_model

        return MLM, AutoTokenizer.from_pretrained(config.pretrained_bert)

    def load_augmentor(self, arg, cal_perplexity=False):
        if isinstance(arg, SentimentClassifier):
            self.sent_classifier = arg
            if hasattr(SentimentClassifier, 'MLM') and hasattr(SentimentClassifier, 'tokenizer'):
                self.MLM, self.tokenizer = self.sent_classifier.MLM, self.sent_classifier.tokenizer
            else:
                self.MLM, self.tokenizer = self.get_mlm_and_tokenizer(self.sent_classifier, self.sent_classifier.opt)
        if not hasattr(self, 'sent_classifier'):
            try:
                self.sent_classifier = APCCheckpointManager.get_sentiment_classifier(arg, cal_perplexity=cal_perplexity, auto_device=self.device)
                self.MLM, self.tokenizer = self.get_mlm_and_tokenizer(self.sent_classifier, self.sent_classifier.opt)
            except:
                keys = ['checkpoint', 'mono_boost', 'deberta', arg]

                checkpoint_path = ''
                max_f1 = ''
                for path in find_dirs(self.ROOT, keys):
                    if 'f1' in path and path[path.index('f1'):] > max_f1:
                        max_f1 = max(path[path.index('f1'):], checkpoint_path)
                        checkpoint_path = path
                if not checkpoint_path:
                    raise ValueError('No trained ckpt found for augmentor initialization, please run augmentation on the target dataset to obtain a ckpt. e.g., BoostAug or MonoAug')
                self.sent_classifier = APCCheckpointManager.get_sentiment_classifier(arg, cal_perplexity=cal_perplexity, auto_device=self.device)
                self.MLM, self.tokenizer = self.get_mlm_and_tokenizer(self.sent_classifier, self.sent_classifier.opt)

    def single_augment(self, text, aspect, label, num=3):

        if self.AUGMENT_BACKEND in 'EDA':
            raw_augs = self.augmenter.augment(text)
        else:
            raw_augs = self.augmenter.augment(text, n=self.AUGMENT_NUM_PER_CASE, num_thread=os.cpu_count())

        if isinstance(raw_augs, str):
            raw_augs = [raw_augs]
        augs = {}
        for text in raw_augs:
            _text = text.replace(aspect, '[ASP]{}[ASP] '.format(aspect))
            _text += '!sent!{}'.format(label)

            with torch.no_grad():
                try:
                    results = self.sent_classifier.infer(_text, print_result=False)
                except:
                    continue
                ids = self.tokenizer(text.replace('PLACEHOLDER', '{}'.format(aspect)), return_tensors="pt")
                ids['labels'] = ids['input_ids'].clone()
                ids = ids.to(self.device)
                loss = self.MLM(**ids)['loss']
                perplexity = torch.exp(loss / ids['input_ids'].size(1))

                if results['ref_check'][0] == 'Correct' and results['confidence'][0] > self.CONFIDENCE_THRESHOLD:
                    augs[perplexity.item()] = [text.replace('PLACEHOLDER', '$T$'), aspect, label]

                augmentations = []
                key_rank = sorted(augs.keys())
                for key in key_rank[:num]:
                    if key < self.PERPLEXITY_THRESHOLD:
                        augmentations += augs[key]

        return augmentations

    def get_apc_config(self, config):
        config.BOOSTING_FOLD = self.BOOSTING_FOLD
        config.CLASSIFIER_TRAINING_NUM = self.CLASSIFIER_TRAINING_NUM
        config.CONFIDENCE_THRESHOLD = self.CONFIDENCE_THRESHOLD
        config.AUGMENT_NUM_PER_CASE = self.AUGMENT_NUM_PER_CASE
        config.WINNER_NUM_PER_CASE = self.WINNER_NUM_PER_CASE
        config.PERPLEXITY_THRESHOLD = self.PERPLEXITY_THRESHOLD
        config.AUGMENT_PCT = self.AUGMENT_PCT
        config.AUGMENT_TOOL = self.AUGMENT_BACKEND
        config.BoostAugVersion = __version__

        apc_config_english = APCConfigManager.get_apc_config_english()
        apc_config_english.cache_dataset = False
        apc_config_english.patience = 10
        apc_config_english.log_step = -1
        apc_config_english.model = APCModelList.FAST_LCF_BERT
        apc_config_english.pretrained_bert = 'microsoft/deberta-v3-base'
        apc_config_english.SRD = 3
        apc_config_english.lcf = 'cdw'
        apc_config_english.optimizer = 'adamw'
        apc_config_english.use_bert_spc = True
        apc_config_english.learning_rate = 1e-5
        apc_config_english.batch_size = 16
        apc_config_english.num_epoch = 25
        apc_config_english.log_step = -1
        apc_config_english.evaluate_begin = 0
        apc_config_english.l2reg = 1e-8
        apc_config_english.cross_validate_fold = -1  # disable cross_validate
        apc_config_english.seed = [random.randint(0, 10000) for _ in range(self.CLASSIFIER_TRAINING_NUM)]
        return apc_config_english

    def apc_classic_augment(self, config: ConfigManager,
                            dataset: DatasetItem,
                            task='apc',
                            rewrite_cache=True,
                            train_after_aug=False
                            ):
        if not isinstance(dataset, DatasetItem):
            dataset = DatasetItem(dataset)
        _config = self.get_apc_config(config)
        tag = '{}_{}_{}'.format(_config.model.__name__.lower(), dataset.dataset_name, os.path.basename(_config.pretrained_bert))

        # if 'lstm' not in config.model.__name__.lower():
        #     tag = '{}_{}_{}'.format(_config.model.__name__.lower(), dataset.dataset_name, os.path.basename(_config.pretrained_bert))
        # else:
        #     tag = '{}_{}_{}'.format(_config.model.__name__.lower(), dataset.dataset_name, 'glove')
        if rewrite_cache:
            prepare_dataset_and_clean_env(dataset.dataset_name, task, rewrite_cache)

        train_data = []
        for dataset_file in detect_dataset(dataset, task)['train']:
            print('processing {}'.format(dataset_file))
            fin = open(dataset_file, encoding='utf8', mode='r')
            lines = fin.readlines()
            fin.close()
            # rename(dataset_file, dataset_file + '.ignore')
            for i in tqdm.tqdm(range(0, len(lines), 3)):
                lines[i] = lines[i].strip()
                lines[i + 1] = lines[i + 1].strip()
                lines[i + 2] = lines[i + 2].strip()

                train_data.append([lines[i], lines[i + 1], lines[i + 2]])

        if self.WINNER_NUM_PER_CASE:

            fout_aug_train = open('{}/classic.train.{}.augment'.format(os.path.dirname(dataset_file), tag), encoding='utf8', mode='w')

            for item in tqdm.tqdm(train_data, postfix='Augmenting...'):

                item[0] = item[0].replace('$T$', 'PLACEHOLDER')

                if self.AUGMENT_BACKEND in 'EDA':
                    augs = self.augmenter.augment(item[0])
                else:
                    augs = self.augmenter.augment(item[0], n=self.AUGMENT_NUM_PER_CASE, num_thread=os.cpu_count())

                if isinstance(augs, str):
                    augs = [augs]
                for aug in augs:
                    if 'PLACEHOLDER' in aug:
                        _text = aug.replace('PLACEHOLDER', '$T$')
                        fout_aug_train.write(_text + '\n')
                        fout_aug_train.write(item[1] + '\n')
                        fout_aug_train.write(item[2] + '\n')

            fout_aug_train.close()

        post_clean(os.path.dirname(dataset_file))

        if train_after_aug:
            print(colored('Start classic augment training...', 'cyan'))
            return Trainer(config=config,
                           dataset=dataset,  # train set and test set will be automatically detected
                           auto_device=self.device  # automatic choose CUDA or CPU
                           ).load_trained_model()

    def apc_boost_augment(self, config: ConfigManager,
                          dataset: DatasetItem,
                          rewrite_cache=True,
                          task='apc',
                          train_after_aug=False
                          ):
        if not isinstance(dataset, DatasetItem):
            dataset = DatasetItem(dataset)
        _config = self.get_apc_config(config)
        tag = '{}_{}_{}'.format(_config.model.__name__.lower(), dataset.dataset_name, os.path.basename(_config.pretrained_bert))

        prepare_dataset_and_clean_env(dataset.dataset_name, task, rewrite_cache)

        for valid_file in detect_dataset(dataset, task)['valid']:
            rename(valid_file, valid_file + '.ignore')

        data = []
        dataset_file = ''
        dataset_files = detect_dataset(dataset, task)['train']

        for dataset_file in dataset_files:
            print('processing {}'.format(dataset_file))
            fin = open(dataset_file, encoding='utf8', mode='r')
            lines = fin.readlines()
            fin.close()
            rename(dataset_file, dataset_file + '.ignore')
            for i in tqdm.tqdm(range(0, len(lines), 3)):
                lines[i] = lines[i].strip()
                lines[i + 1] = lines[i + 1].strip()
                lines[i + 2] = lines[i + 2].strip()

                data.append([lines[i], lines[i + 1], lines[i + 2]])

        train_data = data
        len_per_fold = len(train_data) // self.BOOSTING_FOLD + 1
        folds = [train_data[i: i + len_per_fold] for i in range(0, len(train_data), len_per_fold)]

        if not os.path.exists('checkpoints/cross_boost/{}'.format(tag)):
            os.makedirs('checkpoints/cross_boost/{}'.format(tag))

        for fold_id, b_idx in enumerate(range(len(folds))):
            print(colored('boosting... No.{} in {} folds'.format(b_idx + 1, self.BOOSTING_FOLD), 'red'))
            # f = find_file(self.ROOT, [tag, '{}.'.format(fold_id), dataset.name, '.augment'])
            # if f:
            #     rename(f, f.replace('.ignore', ''))
            #     continue
            train_data = list(itertools.chain(*[x for i, x in enumerate(folds) if i != b_idx]))
            valid_data = folds[b_idx]

            fout_train = open('{}/train.dat.tmp'.format(os.path.dirname(dataset_file), fold_id), encoding='utf8', mode='w')
            fout_boost = open('{}/valid.dat.tmp'.format(os.path.dirname(dataset_file), fold_id), encoding='utf8', mode='w')
            for case in train_data:
                for line in case:
                    fout_train.write(line + '\n')

            for case in valid_data:
                for line in case:
                    fout_boost.write(line + '\n')

            fout_train.close()
            fout_boost.close()

            keys = ['checkpoint', 'cross_boost', dataset.dataset_name, 'fast_lcf_bert', 'deberta', 'No.{}'.format(b_idx + 1)]
            # keys = ['checkpoint', 'cross_boost', 'fast_lcf_bert', 'deberta', 'No.{}'.format(b_idx + 1)]

            if len(find_dirs(self.ROOT, keys)) < self.CLASSIFIER_TRAINING_NUM + 1:
                # _config.log_step = -1
                Trainer(config=_config,
                        dataset=dataset,  # train set and test set will be automatically detected
                        checkpoint_save_mode=1,
                        path_to_save='checkpoints/cross_boost/{}/No.{}/'.format(tag, b_idx + 1),
                        auto_device=self.device  # automatic choose CUDA or CPU
                        )

            torch.cuda.empty_cache()
            time.sleep(5)

            checkpoint_path = ''
            max_f1 = ''
            for path in find_dirs(self.ROOT, keys):
                if 'f1' in path and path[path.index('f1'):] > max_f1:
                    max_f1 = max(path[path.index('f1'):], checkpoint_path)
                    checkpoint_path = path

            self.sent_classifier = APCCheckpointManager.get_sentiment_classifier(checkpoint_path, auto_device=self.device)

            self.sent_classifier.opt.eval_batch_size = 128

            self.MLM, self.tokenizer = self.get_mlm_and_tokenizer(self.sent_classifier, _config)

            dataset_files = detect_dataset(dataset, task)
            boost_sets = dataset_files['valid']
            augmentations = []
            perplexity_list = []
            confidence_list = []

            for boost_set in boost_sets:
                if self.AUGMENT_NUM_PER_CASE <= 0:
                    continue
                print('Augmenting -> {}'.format(boost_set))
                fin = open(boost_set, encoding='utf8', mode='r')
                lines = fin.readlines()
                fin.close()
                remove(boost_set)
                for i in tqdm.tqdm(range(0, len(lines), 3), postfix='No.{} Augmenting...'.format(b_idx + 1)):

                    lines[i] = lines[i].strip().replace('$T$', 'PLACEHOLDER')
                    lines[i + 1] = lines[i + 1].strip()
                    lines[i + 2] = lines[i + 2].strip()

                    if self.AUGMENT_BACKEND in 'EDA':
                        raw_augs = self.augmenter.augment(lines[i])
                    else:
                        raw_augs = self.augmenter.augment(lines[i], n=self.AUGMENT_NUM_PER_CASE, num_thread=os.cpu_count())

                    if isinstance(raw_augs, str):
                        raw_augs = [raw_augs]
                    augs = {}
                    for text in raw_augs:
                        if 'PLACEHOLDER' in text:
                            _text = text.replace('PLACEHOLDER', '[ASP]{}[ASP] '.format(lines[i + 1])) + ' !sent! {}'.format(lines[i + 2])
                        else:
                            continue

                        with torch.no_grad():
                            try:
                                results = self.sent_classifier.infer(_text, print_result=False)
                            except:
                                continue
                            ids = self.tokenizer(text, return_tensors="pt")
                            ids['labels'] = ids['input_ids'].clone()
                            ids = ids.to(self.device)
                            loss = self.MLM(**ids)['loss']
                            perplexity = torch.exp(loss / ids['input_ids'].size(1))

                            perplexity_list.append(perplexity.item())
                            confidence_list.append(results['confidence'][0])

                            if self.USE_LABEL:
                                if results['ref_check'][0] != 'Correct':
                                    continue

                            if self.USE_CONFIDENCE:
                                if results['confidence'][0] <= self.CONFIDENCE_THRESHOLD:
                                    continue
                            augs[perplexity.item()] = [text.replace('PLACEHOLDER', '$T$'), lines[i + 1], lines[i + 2]]

                    if self.USE_CONFIDENCE:
                        key_rank = sorted(augs.keys())
                    else:
                        key_rank = list(augs.keys())

                    for key in key_rank[:self.WINNER_NUM_PER_CASE]:
                        if self.USE_PERPLEXITY:
                            if key < self.PERPLEXITY_THRESHOLD:
                                augmentations += augs[key]
                        else:
                            augmentations += augs[key]

                            # d = aug_dict.get(results['ref_sentiment'][0], [])
                            # d.append([text.replace('PLACEHOLDER', '$T$'), lines[i + 1], lines[i + 2]])
                            # aug_dict[results['ref_sentiment'][0]] = d
            print('Avg Confidence: {} Max Confidence: {} Min Confidence: {}'.format(np.average(confidence_list), max(confidence_list), min(confidence_list)))
            print('Avg Perplexity: {} Max Perplexity: {} Min Perplexity: {}'.format(np.average(perplexity_list), max(perplexity_list), min(perplexity_list)))

            fout = open('{}/{}.cross_boost.{}.train.augment.ignore'.format(os.path.dirname(dataset_file), fold_id, tag), encoding='utf8', mode='w')
            #
            # min_num = min([len(d) for d in aug_dict.values()])
            # for key, value in aug_dict.items():
            #     # random.shuffle(value)
            #     augmentations += value[:int(len(value)*data_dict[key])]
            #
            # for aug in augmentations:
            #     for line in aug:
            #         fout.write(line + '\n')

            for line in augmentations:
                fout.write(line + '\n')
            fout.close()

            del self.sent_classifier
            del self.MLM

            torch.cuda.empty_cache()
            time.sleep(5)

            post_clean(os.path.dirname(dataset_file))

        for f in find_cwd_files('.augment.ignore'):
            rename(f, f.replace('.augment.ignore', ''))

        if train_after_aug:
            print(colored('Start cross boosting augment...', 'green'))
            return Trainer(config=config,
                           dataset=dataset,  # train set and test set will be automatically detected
                           checkpoint_save_mode=0,  # =None to avoid save model
                           auto_device=self.device  # automatic choose CUDA or CPU
                           )

    def apc_mono_augment(self, config: ConfigManager,
                         dataset: DatasetItem,
                         rewrite_cache=True,
                         task='apc',
                         train_after_aug=False
                         ):
        if not isinstance(dataset, DatasetItem):
            dataset = DatasetItem(dataset)
        _config = self.get_apc_config(config)
        tag = '{}_{}_{}'.format(_config.model.__name__.lower(), dataset.dataset_name, os.path.basename(_config.pretrained_bert))

        prepare_dataset_and_clean_env(dataset.dataset_name, task, rewrite_cache)

        if not os.path.exists('checkpoints/mono_boost/{}'.format(tag)):
            os.makedirs('checkpoints/mono_boost/{}'.format(tag))

        print(colored('Begin mono boosting... ', 'yellow'))
        if self.WINNER_NUM_PER_CASE:

            keys = ['checkpoint', 'mono_boost', 'fast_lcf_bert', dataset.dataset_name, 'deberta']

            if len(find_dirs(self.ROOT, keys)) < self.CLASSIFIER_TRAINING_NUM + 1:
                # _config.log_step = -1
                Trainer(config=_config,
                        dataset=dataset,  # train set and test set will be automatically detected
                        checkpoint_save_mode=1,
                        path_to_save='checkpoints/mono_boost/{}/'.format(tag),
                        auto_device=self.device  # automatic choose CUDA or CPU
                        )

            torch.cuda.empty_cache()
            time.sleep(5)

            checkpoint_path = ''
            max_f1 = ''
            for path in find_dirs(self.ROOT, keys):
                if 'f1' in path and path[path.index('f1'):] > max_f1:
                    max_f1 = max(path[path.index('f1'):], checkpoint_path)
                    checkpoint_path = path

            self.sent_classifier = APCCheckpointManager.get_sentiment_classifier(checkpoint_path, auto_device=self.device)

            self.sent_classifier.opt.eval_batch_size = 128

            self.MLM, self.tokenizer = self.get_mlm_and_tokenizer(self.sent_classifier, _config)

            dataset_files = detect_dataset(dataset, task)
            boost_sets = dataset_files['train']
            augmentations = []
            perplexity_list = []
            confidence_list = []

            for boost_set in boost_sets:
                print('Augmenting -> {}'.format(boost_set))
                fin = open(boost_set, encoding='utf8', mode='r')
                lines = fin.readlines()
                fin.close()
                for i in tqdm.tqdm(range(0, len(lines), 3), postfix='Mono Augmenting...'):

                    lines[i] = lines[i].strip().replace('$T$', 'PLACEHOLDER')
                    lines[i + 1] = lines[i + 1].strip()
                    lines[i + 2] = lines[i + 2].strip()

                    if self.AUGMENT_BACKEND in 'EDA':
                        raw_augs = self.augmenter.augment(lines[i])
                    else:
                        raw_augs = self.augmenter.augment(lines[i], n=self.AUGMENT_NUM_PER_CASE, num_thread=os.cpu_count())

                    if isinstance(raw_augs, str):
                        raw_augs = [raw_augs]
                    augs = {}
                    for text in raw_augs:
                        if 'PLACEHOLDER' in text:
                            _text = text.replace('PLACEHOLDER', '[ASP]{}[ASP] '.format(lines[i + 1])) + ' !sent! {}'.format(lines[i + 2])
                        else:
                            continue

                        with torch.no_grad():
                            try:
                                results = self.sent_classifier.infer(_text, print_result=False)
                            except:
                                continue
                            ids = self.tokenizer(text, return_tensors="pt")
                            ids['labels'] = ids['input_ids'].clone()
                            ids = ids.to(self.device)
                            loss = self.MLM(**ids)['loss']
                            perplexity = torch.exp(loss / ids['input_ids'].size(1))

                            perplexity_list.append(perplexity.item())
                            confidence_list.append(results['confidence'][0])

                            if results['ref_check'][0] == 'Correct' and results['confidence'][0] > self.CONFIDENCE_THRESHOLD:
                                augs[perplexity.item()] = [text.replace('PLACEHOLDER', '$T$'), lines[i + 1], lines[i + 2]]

                    key_rank = sorted(augs.keys())
                    for key in key_rank[:self.WINNER_NUM_PER_CASE]:
                        if key < self.PERPLEXITY_THRESHOLD:
                            augmentations += augs[key]

            print('Avg Confidence: {} Max Confidence: {} Min Confidence: {}'.format(np.average(confidence_list), max(confidence_list), min(confidence_list)))
            print('Avg Perplexity: {} Max Perplexity: {} Min Perplexity: {}'.format(np.average(perplexity_list), max(perplexity_list), min(perplexity_list)))

            fout = open('{}/{}.mono_boost.train.augment'.format(os.path.dirname(boost_set), tag), encoding='utf8', mode='w')

            for line in augmentations:
                fout.write(line + '\n')
            fout.close()

            del self.sent_classifier
            del self.MLM

            torch.cuda.empty_cache()
            time.sleep(5)

            post_clean(os.path.dirname(boost_set))

        for f in find_cwd_files('.augment.ignore'):
            rename(f, f.replace('.augment.ignore', ''))

        if train_after_aug:
            print(colored('Start mono boosting augment...', 'yellow'))
            return Trainer(config=config,
                           dataset=dataset,  # train set and test set will be automatically detected
                           checkpoint_save_mode=0,  # =None to avoid save model
                           auto_device=self.device  # automatic choose CUDA or CPU
                           )


class TCBoostAug:

    def __init__(self,
                 ROOT: str = '',
                 BOOSTING_FOLD=5,
                 CLASSIFIER_TRAINING_NUM=2,
                 CONFIDENCE_THRESHOLD=0.99,
                 AUGMENT_NUM_PER_CASE=10,
                 WINNER_NUM_PER_CASE=10,
                 PERPLEXITY_THRESHOLD=4,
                 AUGMENT_PCT=0.1,
                 AUGMENT_BACKEND=AugmentBackend.EDA,
                 USE_CONFIDENCE=True,
                 USE_PERPLEXITY=True,
                 USE_LABEL=True,
                 device='cuda'
                 ):
        """

        :param ROOT: The path to save intermediate checkpoint
        :param BOOSTING_FOLD: Number of splits in crossing boosting augment
        :param CLASSIFIER_TRAINING_NUM: Number of pre-trained inference model using for confidence calculation
        :param CONFIDENCE_THRESHOLD: Confidence threshold used for augmentations filtering
        :param AUGMENT_NUM_PER_CASE: Number of augmentations per example
        :param WINNER_NUM_PER_CASE: Number of selected augmentations per example in confidence ranking
        :param PERPLEXITY_THRESHOLD: Perplexity threshold used for augmentations filtering
        :param AUGMENT_PCT: Word change probability used in backend augment method
        :param AUGMENT_BACKEND: Augmentation backend used for augmentations generation, e.g., EDA, ContextualWordEmbsAug
        """

        assert hasattr(AugmentBackend, AUGMENT_BACKEND)
        if not ROOT or not os.path.exists(ROOT):
            self.ROOT = os.getenv('$HOME') if os.getenv('$HOME') else os.getcwd()
        else:
            self.ROOT = ROOT

        self.BOOSTING_FOLD = BOOSTING_FOLD
        self.CLASSIFIER_TRAINING_NUM = CLASSIFIER_TRAINING_NUM
        self.CONFIDENCE_THRESHOLD = CONFIDENCE_THRESHOLD
        self.AUGMENT_NUM_PER_CASE = AUGMENT_NUM_PER_CASE if AUGMENT_NUM_PER_CASE > 0 else 1
        self.WINNER_NUM_PER_CASE = WINNER_NUM_PER_CASE
        self.PERPLEXITY_THRESHOLD = PERPLEXITY_THRESHOLD
        self.AUGMENT_PCT = AUGMENT_PCT
        self.AUGMENT_BACKEND = AUGMENT_BACKEND
        self.USE_CONFIDENCE = USE_CONFIDENCE
        self.USE_PERPLEXITY = USE_PERPLEXITY
        self.USE_LABEL = USE_LABEL
        self.device = device

        if self.AUGMENT_BACKEND in 'EDA':
            # Here are some augmenters from https://github.com/QData/TextAttack
            from textattack.augmentation import EasyDataAugmenter as Aug
            # Alter default values if desired
            self.augmenter = Aug(pct_words_to_swap=self.AUGMENT_PCT, transformations_per_example=self.AUGMENT_NUM_PER_CASE)
        else:
            # Here are some augmenters from https://github.com/makcedward/nlpaug
            import nlpaug.augmenter.word as naw
            if self.AUGMENT_BACKEND in 'ContextualWordEmbsAug':
                self.augmenter = naw.ContextualWordEmbsAug(
                    model_path='roberta-base', action="substitute", aug_p=self.AUGMENT_PCT, device=self.device)
            elif self.AUGMENT_BACKEND in 'RandomWordAug':
                self.augmenter = naw.RandomWordAug(action="swap")
            elif self.AUGMENT_BACKEND in 'AntonymAug':
                self.augmenter = naw.AntonymAug()
            elif self.AUGMENT_BACKEND in 'SplitAug':
                self.augmenter = naw.SplitAug()
            elif self.AUGMENT_BACKEND in 'BackTranslationAug':
                self.augmenter = naw.BackTranslationAug(from_model_name='facebook/wmt19-en-de',
                                                        to_model_name='facebook/wmt19-de-en',
                                                        device=self.device
                                                        )
            elif self.AUGMENT_BACKEND in 'SpellingAug':
                self.augmenter = naw.SpellingAug()

    def get_mlm_and_tokenizer(self, text_classifier, config):

        if isinstance(text_classifier, TextClassifier):
            base_model = text_classifier.model.bert.base_model
        else:
            base_model = text_classifier.bert.base_model
        pretrained_config = AutoConfig.from_pretrained(config.pretrained_bert)
        try:
            if 'deberta-v3' in config.pretrained_bert:
                MLM = DebertaV2ForMaskedLM(pretrained_config).to(text_classifier.opt.device)
                MLM.deberta = base_model
            elif 'roberta' in config.pretrained_bert:
                MLM = RobertaForMaskedLM(pretrained_config).to(text_classifier.opt.device)
                MLM.roberta = base_model
            else:
                MLM = BertForMaskedLM(pretrained_config).to(text_classifier.opt.device)
                MLM.bert = base_model
        except Exception as e:
            self.device = auto_cuda()
            if 'deberta-v3' in config.pretrained_bert:
                MLM = DebertaV2ForMaskedLM(pretrained_config).to(self.device)
                MLM.deberta = base_model
            elif 'roberta' in config.pretrained_bert:
                MLM = RobertaForMaskedLM(pretrained_config).to(self.device)
                MLM.roberta = base_model
            else:
                MLM = BertForMaskedLM(pretrained_config).to(self.device)
                MLM.bert = base_model

        return MLM, AutoTokenizer.from_pretrained(config.pretrained_bert)

    def load_augmentor(self, arg, cal_perplexity=False):
        if isinstance(arg, TextClassifier):
            self.text_classifier = arg
            if hasattr(TextClassifier, 'MLM') and hasattr(TextClassifier, 'tokenizer'):
                self.MLM, self.tokenizer = self.text_classifier.MLM, self.text_classifier.tokenizer
            else:
                self.MLM, self.tokenizer = self.get_mlm_and_tokenizer(self.text_classifier, self.text_classifier.opt)
        if not hasattr(self, 'text_classifier'):
            try:
                self.text_classifier = TCCheckpointManager.get_text_classifier(arg, cal_perplexity=cal_perplexity, auto_device=self.device)
                self.MLM, self.tokenizer = self.get_mlm_and_tokenizer(self.text_classifier, self.text_classifier.opt)
            except:
                keys = ['checkpoint', 'mono_boost', 'deberta', arg]

                checkpoint_path = ''
                max_f1 = ''
                for path in find_dirs(self.ROOT, keys):
                    if 'f1' in path and path[path.index('f1'):] > max_f1:
                        max_f1 = max(path[path.index('f1'):], checkpoint_path)
                        checkpoint_path = path
                if not checkpoint_path:
                    raise ValueError('No trained ckpt found for augmentor initialization, please run augmentation on the target dataset to obtain a ckpt. e.g., BoostAug or MonoAug')
                self.text_classifier = TCCheckpointManager.get_text_classifier(arg, cal_perplexity=cal_perplexity, auto_device=self.device)
                self.MLM, self.tokenizer = self.get_mlm_and_tokenizer(self.text_classifier, self.text_classifier.opt)

    def single_augment(self, text, label, num=3):

        if self.AUGMENT_BACKEND in 'EDA':
            raw_augs = self.augmenter.augment(text)
        else:
            raw_augs = self.augmenter.augment(text, n=self.AUGMENT_NUM_PER_CASE, num_thread=os.cpu_count())

        if isinstance(raw_augs, str):
            raw_augs = [raw_augs]
        augs = {}
        for text in raw_augs:
            with torch.no_grad():
                try:
                    results = self.text_classifier.infer(text + '!ref!{}'.format(label), print_result=False)
                except:
                    continue
                ids = self.tokenizer(text, return_tensors="pt")
                ids['labels'] = ids['input_ids'].clone()
                ids = ids.to(self.device)
                loss = self.MLM(**ids)['loss']
                perplexity = torch.exp(loss / ids['input_ids'].size(1))

                if self.USE_LABEL:
                    if results['ref_check'] != 'Correct':
                        continue

                if self.USE_CONFIDENCE:
                    if results['confidence'] <= self.CONFIDENCE_THRESHOLD:
                        continue

                augs[perplexity.item()] = [text.replace('PLACEHOLDER', '$LABEL$')]

        if self.USE_CONFIDENCE:
            # key_rank = list(reversed(sorted(augs.keys())))
            key_rank = sorted(augs.keys())
        else:
            key_rank = list(augs.keys())
        augmentations = []
        for key in key_rank[:num]:
            if self.USE_PERPLEXITY:
                if key < self.PERPLEXITY_THRESHOLD:
                    augmentations += augs[key]

        return augmentations

    def get_tc_config(self, config):
        config.BOOSTING_FOLD = self.BOOSTING_FOLD
        config.CLASSIFIER_TRAINING_NUM = self.CLASSIFIER_TRAINING_NUM
        config.CONFIDENCE_THRESHOLD = self.CONFIDENCE_THRESHOLD
        config.AUGMENT_NUM_PER_CASE = self.AUGMENT_NUM_PER_CASE
        config.WINNER_NUM_PER_CASE = self.WINNER_NUM_PER_CASE
        config.PERPLEXITY_THRESHOLD = self.PERPLEXITY_THRESHOLD
        config.AUGMENT_PCT = self.AUGMENT_PCT
        config.AUGMENT_TOOL = self.AUGMENT_BACKEND
        config.BoostAugVersion = __version__
        tc_config_english = TCConfigManager.get_tc_config_english()
        tc_config_english.max_seq_len = 80
        tc_config_english.dropout = 0
        tc_config_english.model = BERTTCModelList.BERT
        tc_config_english.pretrained_bert = 'microsoft/deberta-v3-base'
        tc_config_english.optimizer = 'adamw'
        tc_config_english.cache_dataset = False
        tc_config_english.patience = 10
        tc_config_english.log_step = -1
        tc_config_english.learning_rate = 1e-5
        tc_config_english.batch_size = 16
        tc_config_english.num_epoch = 10
        tc_config_english.evaluate_begin = 0
        tc_config_english.l2reg = 1e-8
        tc_config_english.cross_validate_fold = -1  # disable cross_validate
        tc_config_english.seed = [random.randint(0, 10000) for _ in range(self.CLASSIFIER_TRAINING_NUM)]
        return tc_config_english

    def tc_classic_augment(self, config: ConfigManager,
                           dataset: DatasetItem,
                           rewrite_cache=True,
                           task='text_classification',
                           train_after_aug=False
                           ):
        if not isinstance(dataset, DatasetItem):
            dataset = DatasetItem(dataset)
        _config = self.get_tc_config(config)
        tag = '{}_{}_{}'.format(_config.model.__name__.lower(), dataset.dataset_name, os.path.basename(_config.pretrained_bert))
        if rewrite_cache:
            prepare_dataset_and_clean_env(dataset.dataset_name, task, rewrite_cache)

        train_data = []
        for dataset_file in detect_dataset(dataset, task)['train']:
            print('processing {}'.format(dataset_file))
            fin = open(dataset_file, encoding='utf8', mode='r')
            lines = fin.readlines()
            fin.close()
            for i in tqdm.tqdm(range(0, len(lines))):
                lines[i] = lines[i].strip()
                train_data.append([lines[i]])
        fs = find_files(self.ROOT, [tag, '.augment.ignore'])
        if self.WINNER_NUM_PER_CASE:

            fout_aug_train = open('{}/classic.train.{}.augment'.format(os.path.dirname(dataset_file), tag), encoding='utf8', mode='w')

            for item in tqdm.tqdm(train_data, postfix='Classic Augmenting...'):

                item[0] = item[0].replace('$LABEL$', 'PLACEHOLDER')
                label = item[0].split('PLACEHOLDER')[1].strip()

                if self.AUGMENT_BACKEND in 'EDA':
                    augs = self.augmenter.augment(item[0])
                else:
                    augs = self.augmenter.augment(item[0], n=self.AUGMENT_NUM_PER_CASE, num_thread=os.cpu_count())

                if isinstance(augs, str):
                    augs = [augs]
                for aug in augs:
                    if aug.endswith('PLACEHOLDER {}'.format(label)) or aug.endswith('PLACEHOLDER{}'.format(label)):
                        _text = aug.replace('PLACEHOLDER', '$LABEL$')
                        fout_aug_train.write(_text + '\n')

            fout_aug_train.close()

        post_clean(os.path.dirname(dataset_file))

        if train_after_aug:
            print(colored('Start classic augment training...', 'cyan'))
            return Trainer(config=config,
                           dataset=dataset,  # train set and test set will be automatically detected
                           auto_device=self.device  # automatic choose CUDA or CPU
                           ).load_trained_model()

    def tc_boost_augment(self, config: ConfigManager,
                         dataset: DatasetItem,
                         rewrite_cache=True,
                         task='text_classification',
                         train_after_aug=False
                         ):
        if not isinstance(dataset, DatasetItem):
            dataset = DatasetItem(dataset)
        _config = self.get_tc_config(config)
        tag = '{}_{}_{}'.format(_config.model.__name__.lower(), dataset.dataset_name, os.path.basename(_config.pretrained_bert))

        prepare_dataset_and_clean_env(dataset.dataset_name, task, rewrite_cache)

        for valid_file in detect_dataset(dataset, task)['valid']:
            rename(valid_file, valid_file + '.ignore')

        data = []
        dataset_file = ''
        dataset_files = detect_dataset(dataset, task)['train']

        for dataset_file in dataset_files:
            print('processing {}'.format(dataset_file))
            fin = open(dataset_file, encoding='utf8', mode='r')
            lines = fin.readlines()
            fin.close()
            rename(dataset_file, dataset_file + '.ignore')
            for i in tqdm.tqdm(range(0, len(lines))):
                lines[i] = lines[i].strip()

                data.append([lines[i]])

        train_data = data
        len_per_fold = len(train_data) // self.BOOSTING_FOLD + 1
        folds = [train_data[i: i + len_per_fold] for i in range(0, len(train_data), len_per_fold)]

        if not os.path.exists('checkpoints/cross_boost/{}_{}'.format(config.model.__name__.lower(), dataset.dataset_name)):
            os.makedirs('checkpoints/cross_boost/{}_{}'.format(config.model.__name__.lower(), dataset.dataset_name))

        for fold_id, b_idx in enumerate(range(len(folds))):
            print(colored('boosting... No.{} in {} folds'.format(b_idx + 1, self.BOOSTING_FOLD), 'red'))
            # f = find_file(self.ROOT, [tag, '{}.'.format(fold_id), dataset.name, '.augment'])
            # if f:
            #     rename(f, f.replace('.ignore', ''))
            #     continue
            train_data = list(itertools.chain(*[x for i, x in enumerate(folds) if i != b_idx]))
            valid_data = folds[b_idx]

            fout_train = open('{}/train.dat.tmp'.format(os.path.dirname(dataset_file), fold_id), encoding='utf8', mode='w')
            fout_boost = open('{}/valid.dat.tmp'.format(os.path.dirname(dataset_file), fold_id), encoding='utf8', mode='w')
            for case in train_data:
                for line in case:
                    fout_train.write(line + '\n')

            for case in valid_data:
                for line in case:
                    fout_boost.write(line + '\n')

            fout_train.close()
            fout_boost.close()

            keys = ['checkpoint', 'cross_boost', dataset.dataset_name, 'deberta', 'No.{}'.format(b_idx + 1)]

            if len(find_dirs(self.ROOT, keys)) < self.CLASSIFIER_TRAINING_NUM + 1:
                Trainer(config=_config,
                        dataset=dataset,  # train set and test set will be automatically detected
                        checkpoint_save_mode=1,
                        path_to_save='checkpoints/cross_boost/{}/No.{}'.format(tag, b_idx + 1),
                        auto_device=self.device  # automatic choose CUDA or CPU
                        )

            torch.cuda.empty_cache()
            time.sleep(5)

            checkpoint_path = ''
            max_f1 = ''
            for path in find_dirs(self.ROOT, keys):
                if 'f1' in path and path[path.index('f1'):] > max_f1:
                    max_f1 = max(path[path.index('f1'):], checkpoint_path)
                    checkpoint_path = path

            self.text_classifier = TCCheckpointManager.get_text_classifier(checkpoint_path, auto_device=self.device)
            self.text_classifier.opt.eval_batch_size = 128

            self.MLM, self.tokenizer = self.get_mlm_and_tokenizer(self.text_classifier, _config)

            dataset_files = detect_dataset(dataset, task)
            boost_sets = dataset_files['valid']
            augmentations = []
            perplexity_list = []
            confidence_list = []

            for boost_set in boost_sets:
                print('Augmenting -> {}'.format(boost_set))
                fin = open(boost_set, encoding='utf8', mode='r')
                lines = fin.readlines()
                fin.close()
                remove(boost_set)
                for i in tqdm.tqdm(range(0, len(lines)), postfix='No.{} Augmenting...'.format(b_idx + 1)):

                    lines[i] = lines[i].strip().replace('$LABEL$', 'PLACEHOLDER')
                    label = lines[i].split('PLACEHOLDER')[1].strip()

                    if self.AUGMENT_BACKEND in 'EDA':
                        raw_augs = self.augmenter.augment(lines[i])
                    else:
                        raw_augs = self.augmenter.augment(lines[i], n=self.AUGMENT_NUM_PER_CASE, num_thread=os.cpu_count())

                    if isinstance(raw_augs, str):
                        raw_augs = [raw_augs]
                    augs = {}
                    for text in raw_augs:
                        if text.endswith('PLACEHOLDER {}'.format(label)) or text.endswith('PLACEHOLDER{}'.format(label)):
                            with torch.no_grad():
                                try:
                                    results = self.text_classifier.infer(text.replace('PLACEHOLDER', '!ref!'), print_result=False)
                                except:
                                    continue
                                ids = self.tokenizer(text, return_tensors="pt")
                                ids['labels'] = ids['input_ids'].clone()
                                ids = ids.to(self.device)
                                loss = self.MLM(**ids)['loss']
                                perplexity = torch.exp(loss / ids['input_ids'].size(1))

                                perplexity_list.append(perplexity.item())
                                confidence_list.append(results['confidence'])
                                if self.USE_LABEL:
                                    if results['ref_check'] != 'Correct':
                                        continue

                                if self.USE_CONFIDENCE:
                                    if results['confidence'] <= self.CONFIDENCE_THRESHOLD:
                                        continue

                                augs[perplexity.item()] = [text.replace('PLACEHOLDER', '$LABEL$')]

                    if self.USE_CONFIDENCE:
                        key_rank = sorted(augs.keys())
                    else:
                        key_rank = list(augs.keys())
                    for key in key_rank[:self.WINNER_NUM_PER_CASE]:
                        if self.USE_PERPLEXITY:
                            if key < self.PERPLEXITY_THRESHOLD:
                                augmentations += augs[key]
                        else:
                            augmentations += augs[key]

            print('Avg Confidence: {} Max Confidence: {} Min Confidence: {}'.format(np.average(confidence_list), max(confidence_list), min(confidence_list)))

            print('Avg Perplexity: {} Max Perplexity: {} Min Perplexity: {}'.format(np.average(perplexity_list), max(perplexity_list), min(perplexity_list)))

            fout = open('{}/{}.cross_boost.{}.train.augment.ignore'.format(os.path.dirname(dataset_file), fold_id, tag), encoding='utf8', mode='w')

            for line in augmentations:
                fout.write(line + '\n')
            fout.close()

            del self.text_classifier
            del self.MLM

            torch.cuda.empty_cache()
            time.sleep(5)

            post_clean(os.path.dirname(dataset_file))

        for f in find_cwd_files('.augment.ignore'):
            rename(f, f.replace('.augment.ignore', ''))

        if train_after_aug:
            print(colored('Start cross boosting augment...', 'green'))
            return Trainer(config=config,
                           dataset=dataset,  # train set and test set will be automatically detected
                           checkpoint_save_mode=0,  # =None to avoid save model
                           auto_device=self.device  # automatic choose CUDA or CPU
                           )

    def tc_mono_augment(self, config: ConfigManager,
                        dataset: DatasetItem,
                        rewrite_cache=True,
                        task='text_classification',
                        train_after_aug=False
                        ):
        if not isinstance(dataset, DatasetItem):
            dataset = DatasetItem(dataset)
        _config = self.get_tc_config(config)
        tag = '{}_{}_{}'.format(_config.model.__name__.lower(), dataset.dataset_name, os.path.basename(_config.pretrained_bert))

        prepare_dataset_and_clean_env(dataset.dataset_name, task, rewrite_cache)

        if not os.path.exists('checkpoints/mono_boost/{}'.format(tag)):
            os.makedirs('checkpoints/mono_boost/{}'.format(tag))

        print(colored('Begin mono boosting... ', 'yellow'))
        if self.WINNER_NUM_PER_CASE:

            keys = ['checkpoint', 'mono_boost', dataset.dataset_name, 'deberta']

            if len(find_dirs(self.ROOT, keys)) < self.CLASSIFIER_TRAINING_NUM + 1:
                # _config.log_step = -1
                Trainer(config=_config,
                        dataset=dataset,  # train set and test set will be automatically detected
                        checkpoint_save_mode=1,
                        path_to_save='checkpoints/mono_boost/{}/'.format(tag),
                        auto_device=self.device  # automatic choose CUDA or CPU
                        )

            torch.cuda.empty_cache()
            time.sleep(5)

            checkpoint_path = ''
            max_f1 = ''
            for path in find_dirs(self.ROOT, keys):
                if 'f1' in path and path[path.index('f1'):] > max_f1:
                    max_f1 = max(path[path.index('f1'):], checkpoint_path)
                    checkpoint_path = path

            self.text_classifier = TCCheckpointManager.get_text_classifier(checkpoint_path, cal_perplexity=True, auto_device=self.device)

            self.text_classifier.opt.eval_batch_size = 128

            self.MLM, self.tokenizer = self.get_mlm_and_tokenizer(self.text_classifier, _config)

            dataset_files = detect_dataset(dataset, task)
            boost_sets = dataset_files['train']
            augmentations = []
            perplexity_list = []
            confidence_list = []

            for boost_set in boost_sets:
                print('Augmenting -> {}'.format(boost_set))
                fin = open(boost_set, encoding='utf8', mode='r')
                lines = fin.readlines()
                fin.close()
                # remove(boost_set)
                for i in tqdm.tqdm(range(0, len(lines)), postfix='Mono Augmenting...'):

                    lines[i] = lines[i].strip().replace('$LABEL$', 'PLACEHOLDER')
                    label = lines[i].split('PLACEHOLDER')[1].strip()

                    if self.AUGMENT_BACKEND in 'EDA':
                        raw_augs = self.augmenter.augment(lines[i])
                    else:
                        raw_augs = self.augmenter.augment(lines[i], n=self.AUGMENT_NUM_PER_CASE, num_thread=os.cpu_count())

                    if isinstance(raw_augs, str):
                        raw_augs = [raw_augs]
                    augs = {}
                    for text in raw_augs:
                        if text.endswith('PLACEHOLDER {}'.format(label)) or text.endswith('PLACEHOLDER{}'.format(label)):
                            with torch.no_grad():
                                try:
                                    results = self.text_classifier.infer(text.replace('PLACEHOLDER', '!ref!'), print_result=False)
                                except:
                                    continue
                                ids = self.tokenizer(text, return_tensors="pt")
                                ids['labels'] = ids['input_ids'].clone()
                                ids = ids.to(self.device)
                                loss = self.MLM(**ids)['loss']
                                perplexity = torch.exp(loss / ids['input_ids'].size(1))

                                perplexity_list.append(perplexity.item())
                                confidence_list.append(results['confidence'])

                                if results['ref_check'] == 'Correct' and results['confidence'] > self.CONFIDENCE_THRESHOLD:
                                    augs[perplexity.item()] = [text.replace('PLACEHOLDER', '$LABEL$')]

                    key_rank = sorted(augs.keys())
                    for key in key_rank[:self.WINNER_NUM_PER_CASE]:
                        if key < self.PERPLEXITY_THRESHOLD:
                            augmentations += augs[key]

            print('Avg Confidence: {} Max Confidence: {} Min Confidence: {}'.format(np.average(confidence_list), max(confidence_list), min(confidence_list)))

            print('Avg Perplexity: {} Max Perplexity: {} Min Perplexity: {}'.format(np.average(perplexity_list), max(perplexity_list), min(perplexity_list)))

            fout = open('{}/{}.mono_boost.train.augment.ignore'.format(os.path.dirname(boost_set), tag), encoding='utf8', mode='w')

            for line in augmentations:
                fout.write(line + '\n')
            fout.close()

            del self.text_classifier
            del self.MLM

            torch.cuda.empty_cache()
            time.sleep(5)

            post_clean(os.path.dirname(boost_set))

        for f in find_cwd_files('.augment.ignore'):
            rename(f, f.replace('.augment.ignore', ''))

        if train_after_aug:
            print(colored('Start mono boosting augment...', 'yellow'))
            return Trainer(config=config,
                           dataset=dataset,  # train set and test set will be automatically detected
                           checkpoint_save_mode=0,  # =None to avoid save model
                           auto_device=self.device  # automatic choose CUDA or CPU
                           )


class TADBoostAug:

    def __init__(self,
                 ROOT: str = '',
                 BOOSTING_FOLD=5,
                 CLASSIFIER_TRAINING_NUM=2,
                 CONFIDENCE_THRESHOLD=0.99,
                 AUGMENT_NUM_PER_CASE=10,
                 WINNER_NUM_PER_CASE=10,
                 PERPLEXITY_THRESHOLD=4,
                 AUGMENT_PCT=0.1,
                 AUGMENT_BACKEND=AugmentBackend.EDA,
                 USE_CONFIDENCE=True,
                 USE_PERPLEXITY=True,
                 USE_LABEL=True,
                 device='cuda'
                 ):
        """

        :param ROOT: The path to save intermediate checkpoint
        :param BOOSTING_FOLD: Number of splits in crossing boosting augment
        :param CLASSIFIER_TRAINING_NUM: Number of pre-trained inference model using for confidence calculation
        :param CONFIDENCE_THRESHOLD: Confidence threshold used for augmentations filtering
        :param AUGMENT_NUM_PER_CASE: Number of augmentations per example
        :param WINNER_NUM_PER_CASE: Number of selected augmentations per example in confidence ranking
        :param PERPLEXITY_THRESHOLD: Perplexity threshold used for augmentations filtering
        :param AUGMENT_PCT: Word change probability used in backend augment method
        :param AUGMENT_BACKEND: Augmentation backend used for augmentations generation, e.g., EDA, ContextualWordEmbsAug
        """

        assert hasattr(AugmentBackend, AUGMENT_BACKEND)
        if not ROOT or not os.path.exists(ROOT):
            self.ROOT = os.getenv('$HOME') if os.getenv('$HOME') else os.getcwd()
        else:
            self.ROOT = ROOT

        self.BOOSTING_FOLD = BOOSTING_FOLD
        self.CLASSIFIER_TRAINING_NUM = CLASSIFIER_TRAINING_NUM
        self.CONFIDENCE_THRESHOLD = CONFIDENCE_THRESHOLD
        self.AUGMENT_NUM_PER_CASE = AUGMENT_NUM_PER_CASE if AUGMENT_NUM_PER_CASE > 0 else 1
        self.WINNER_NUM_PER_CASE = WINNER_NUM_PER_CASE
        self.PERPLEXITY_THRESHOLD = PERPLEXITY_THRESHOLD
        self.AUGMENT_PCT = AUGMENT_PCT
        self.AUGMENT_BACKEND = AUGMENT_BACKEND
        self.USE_CONFIDENCE = USE_CONFIDENCE
        self.USE_PERPLEXITY = USE_PERPLEXITY
        self.USE_LABEL = USE_LABEL
        self.device = device

        if self.AUGMENT_BACKEND in 'EDA':
            # Here are some augmenters from https://github.com/QData/TextAttack
            from textattack.augmentation import EasyDataAugmenter as Aug
            # Alter default values if desired
            self.augmenter = Aug(pct_words_to_swap=self.AUGMENT_PCT, transformations_per_example=self.AUGMENT_NUM_PER_CASE)
        else:
            # Here are some augmenters from https://github.com/makcedward/nlpaug
            import nlpaug.augmenter.word as naw
            if self.AUGMENT_BACKEND in 'ContextualWordEmbsAug':
                self.augmenter = naw.ContextualWordEmbsAug(
                    model_path='roberta-base', action="substitute", aug_p=self.AUGMENT_PCT, device=self.device)
            elif self.AUGMENT_BACKEND in 'RandomWordAug':
                self.augmenter = naw.RandomWordAug(action="swap")
            elif self.AUGMENT_BACKEND in 'AntonymAug':
                self.augmenter = naw.AntonymAug()
            elif self.AUGMENT_BACKEND in 'SplitAug':
                self.augmenter = naw.SplitAug()
            elif self.AUGMENT_BACKEND in 'BackTranslationAug':
                self.augmenter = naw.BackTranslationAug(from_model_name='facebook/wmt19-en-de',
                                                        to_model_name='facebook/wmt19-de-en',
                                                        device=self.device
                                                        )
            elif self.AUGMENT_BACKEND in 'SpellingAug':
                self.augmenter = naw.SpellingAug()

    def get_mlm_and_tokenizer(self, text_classifier, config):

        if isinstance(text_classifier, TADTextClassifier):
            base_model = text_classifier.model.bert.base_model
        else:
            base_model = text_classifier.bert.base_model
        pretrained_config = AutoConfig.from_pretrained(config.pretrained_bert)
        try:
            if 'deberta-v3' in config.pretrained_bert:
                MLM = DebertaV2ForMaskedLM(pretrained_config).to(text_classifier.opt.device)
                MLM.deberta = base_model
            elif 'roberta' in config.pretrained_bert:
                MLM = RobertaForMaskedLM(pretrained_config).to(text_classifier.opt.device)
                MLM.roberta = base_model
            else:
                MLM = BertForMaskedLM(pretrained_config).to(text_classifier.opt.device)
                MLM.bert = base_model
        except Exception as e:
            self.device = auto_cuda()
            if 'deberta-v3' in config.pretrained_bert:
                MLM = DebertaV2ForMaskedLM(pretrained_config).to(self.device)
                MLM.deberta = base_model
            elif 'roberta' in config.pretrained_bert:
                MLM = RobertaForMaskedLM(pretrained_config).to(self.device)
                MLM.roberta = base_model
            else:
                MLM = BertForMaskedLM(pretrained_config).to(self.device)
                MLM.bert = base_model

        return MLM, AutoTokenizer.from_pretrained(config.pretrained_bert)

    def load_augmentor(self, arg, cal_perplexity=False):
        if isinstance(arg, TADTextClassifier):
            self.tad_classifier = arg
            if hasattr(TADTextClassifier, 'MLM') and hasattr(TADTextClassifier, 'tokenizer'):
                self.MLM, self.tokenizer = self.tad_classifier.MLM, self.tad_classifier.tokenizer
            else:
                self.MLM, self.tokenizer = self.get_mlm_and_tokenizer(self.tad_classifier, self.tad_classifier.opt)
        if not hasattr(self, 'tad_classifier'):
            try:
                self.tad_classifier = TADCheckpointManager.get_tad_text_classifier(arg, cal_perplexity=cal_perplexity, auto_device=self.device)
                self.MLM, self.tokenizer = self.get_mlm_and_tokenizer(self.tad_classifier, self.tad_classifier.opt)
            except:
                keys = ['checkpoint', 'mono_boost', 'deberta', arg]

                checkpoint_path = ''
                max_f1 = ''
                for path in find_dirs(self.ROOT, keys):
                    if 'f1' in path and path[path.index('f1'):] > max_f1:
                        max_f1 = max(path[path.index('f1'):], checkpoint_path)
                        checkpoint_path = path
                if not checkpoint_path:
                    raise ValueError('No trained ckpt found for augmentor initialization, please run augmentation on the target dataset to obtain a ckpt. e.g., BoostAug or MonoAug')
                self.tad_classifier = TADCheckpointManager.get_tad_text_classifier(checkpoint_path, cal_perplexity=cal_perplexity, auto_device=self.device)
                self.MLM, self.tokenizer = self.get_mlm_and_tokenizer(self.tad_classifier, self.tad_classifier.opt)

    def single_augment(self, text, label, num=3):

        if self.AUGMENT_BACKEND in 'EDA':
            raw_augs = self.augmenter.augment(text)
        else:
            raw_augs = self.augmenter.augment(text, n=self.AUGMENT_NUM_PER_CASE, num_thread=os.cpu_count())

        if isinstance(raw_augs, str):
            raw_augs = [raw_augs]
        augs = {}
        for text in raw_augs:
            with torch.no_grad():
                try:
                    results = self.tad_classifier.infer(text + '!ref!{},-100,-100'.format(label), print_result=False, attack_defense=False)
                except Exception as e:
                    raise e
                ids = self.tokenizer(text, return_tensors="pt")
                ids['labels'] = ids['input_ids'].clone()
                ids = ids.to(self.device)
                loss = self.MLM(**ids)['loss']
                perplexity = torch.exp(loss / ids['input_ids'].size(1))

                if self.USE_LABEL:
                    if results['ref_label_check'] != 'Correct':
                        continue

                if self.USE_CONFIDENCE:
                    if results['confidence'] <= self.CONFIDENCE_THRESHOLD:
                        continue

                augs[perplexity.item()] = [text.replace('PLACEHOLDER', '$LABEL$')]

        if self.USE_CONFIDENCE:
            key_rank = sorted(augs.keys())
        else:
            key_rank = list(augs.keys())
        augmentations = []
        for key in key_rank[:num]:
            if self.USE_PERPLEXITY:
                if key < self.PERPLEXITY_THRESHOLD:
                    augmentations += augs[key]

        return augmentations

    def get_tad_config(self, config):
        config.BOOSTING_FOLD = self.BOOSTING_FOLD
        config.CLASSIFIER_TRAINING_NUM = self.CLASSIFIER_TRAINING_NUM
        config.CONFIDENCE_THRESHOLD = self.CONFIDENCE_THRESHOLD
        config.AUGMENT_NUM_PER_CASE = self.AUGMENT_NUM_PER_CASE
        config.WINNER_NUM_PER_CASE = self.WINNER_NUM_PER_CASE
        config.PERPLEXITY_THRESHOLD = self.PERPLEXITY_THRESHOLD
        config.AUGMENT_PCT = self.AUGMENT_PCT
        config.AUGMENT_TOOL = self.AUGMENT_BACKEND
        config.BoostAugVersion = __version__
        tad_config_english = TADConfigManager.get_tad_config_english()
        tad_config_english.max_seq_len = 80
        tad_config_english.dropout = 0
        tad_config_english.model = TADBERT
        tad_config_english.pretrained_bert = 'microsoft/deberta-v3-base'
        tad_config_english.optimizer = 'adamw'
        tad_config_english.cache_dataset = False
        tad_config_english.patience = 10
        tad_config_english.log_step = -1
        tad_config_english.learning_rate = 1e-5
        tad_config_english.batadh_size = 16
        tad_config_english.num_epoch = 10
        tad_config_english.evaluate_begin = 0
        tad_config_english.l2reg = 1e-8
        tad_config_english.cross_validate_fold = -1  # disable cross_validate
        tad_config_english.seed = [random.randint(0, 10000) for _ in range(self.CLASSIFIER_TRAINING_NUM)]
        return tad_config_english

    def tad_classic_augment(self, config: ConfigManager,
                            dataset: DatasetItem,
                            rewrite_cache=True,
                            task='text_defense',
                            train_after_aug=False
                            ):
        if not isinstance(dataset, DatasetItem):
            dataset = DatasetItem(dataset)
        _config = self.get_tad_config(config)
        tag = '{}_{}_{}'.format(_config.model.__name__.lower(), dataset.dataset_name, os.path.basename(_config.pretrained_bert))
        if rewrite_cache:
            prepare_dataset_and_clean_env(dataset.dataset_name, task, rewrite_cache)

        train_data = []
        for dataset_file in detect_dataset(dataset, task)['train']:
            print('processing {}'.format(dataset_file))
            fin = open(dataset_file, encoding='utf8', mode='r')
            lines = fin.readlines()
            fin.close()
            for i in tqdm.tqdm(range(0, len(lines))):
                lines[i] = lines[i].strip()
                train_data.append([lines[i]])

        if self.WINNER_NUM_PER_CASE:

            fout_aug_train = open('{}/classic.train.{}.augment'.format(os.path.dirname(dataset_file), tag), encoding='utf8', mode='w')

            for item in tqdm.tqdm(train_data, postfix='Classic Augmenting...'):

                item[0] = item[0].replace('$LABEL$', 'PLACEHOLDER')
                label = item[0].split('PLACEHOLDER')[1].strip()

                if self.AUGMENT_BACKEND in 'EDA':
                    augs = self.augmenter.augment(item[0])
                else:
                    augs = self.augmenter.augment(item[0], n=self.AUGMENT_NUM_PER_CASE, num_thread=os.cpu_count())

                if isinstance(augs, str):
                    augs = [augs]
                for aug in augs:
                    if aug.endswith('PLACEHOLDER {}'.format(label)) or aug.endswith('PLACEHOLDER{}'.format(label)):
                        _text = aug.replace('PLACEHOLDER', '$LABEL$')
                        fout_aug_train.write(_text + '\n')

            fout_aug_train.close()

        post_clean(os.path.dirname(dataset_file))

        if train_after_aug:
            print(colored('Start classic augment training...', 'cyan'))
            return Trainer(config=config,
                           dataset=dataset,  # train set and test set will be automatically detected
                           auto_device=self.device  # automatic choose CUDA or CPU
                           ).load_trained_model()

    def tad_boost_augment(self, config: ConfigManager,
                          dataset: DatasetItem,
                          rewrite_cache=True,
                          task='text_defense',
                          train_after_aug=False
                          ):
        if not isinstance(dataset, DatasetItem):
            dataset = DatasetItem(dataset)
        _config = self.get_tad_config(config)
        tag = '{}_{}_{}'.format(_config.model.__name__.lower(), dataset.dataset_name, os.path.basename(_config.pretrained_bert))

        prepare_dataset_and_clean_env(dataset.dataset_name, task, rewrite_cache)

        for valid_file in detect_dataset(dataset, task)['valid']:
            rename(valid_file, valid_file + '.ignore')

        data = []
        dataset_file = ''
        dataset_files = detect_dataset(dataset, task)['train']

        for dataset_file in dataset_files:
            print('processing {}'.format(dataset_file))
            fin = open(dataset_file, encoding='utf8', mode='r')
            lines = fin.readlines()
            fin.close()
            rename(dataset_file, dataset_file + '.ignore')
            for i in tqdm.tqdm(range(0, len(lines))):
                lines[i] = lines[i].strip()

                data.append([lines[i]])

        train_data = data
        len_per_fold = len(train_data) // self.BOOSTING_FOLD + 1
        folds = [train_data[i: i + len_per_fold] for i in range(0, len(train_data), len_per_fold)]

        if not os.path.exists('checkpoints/cross_boost/{}_{}'.format(config.model.__name__.lower(), dataset.dataset_name)):
            os.makedirs('checkpoints/cross_boost/{}_{}'.format(config.model.__name__.lower(), dataset.dataset_name))

        for fold_id, b_idx in enumerate(range(len(folds))):
            print(colored('boosting... No.{} in {} folds'.format(b_idx + 1, self.BOOSTING_FOLD), 'red'))
            train_data = list(itertools.chain(*[x for i, x in enumerate(folds) if i != b_idx]))
            valid_data = folds[b_idx]

            fout_train = open('{}/train.dat.tmp'.format(os.path.dirname(dataset_file), fold_id), encoding='utf8', mode='w')
            fout_boost = open('{}/valid.dat.tmp'.format(os.path.dirname(dataset_file), fold_id), encoding='utf8', mode='w')
            for case in train_data:
                for line in case:
                    fout_train.write(line + '\n')

            for case in valid_data:
                for line in case:
                    fout_boost.write(line + '\n')

            fout_train.close()
            fout_boost.close()

            keys = ['checkpoint', 'cross_boost', dataset.dataset_name, 'deberta', 'No.{}'.format(b_idx + 1)]

            if len(find_dirs(self.ROOT, keys)) < self.CLASSIFIER_TRAINING_NUM + 1:
                Trainer(config=_config,
                        dataset=dataset,  # train set and test set will be automatically detected
                        checkpoint_save_mode=1,
                        path_to_save='checkpoints/cross_boost/{}/No.{}'.format(tag, b_idx + 1),
                        auto_device=self.device  # automatic choose CUDA or CPU
                        )

            torch.cuda.empty_cache()
            time.sleep(5)

            checkpoint_path = ''
            max_f1 = ''
            for path in find_dirs(self.ROOT, keys):
                if 'f1' in path and path[path.index('f1'):] > max_f1:
                    max_f1 = max(path[path.index('f1'):], checkpoint_path)
                    checkpoint_path = path

            self.tad_classifier = TADCheckpointManager.get_tad_text_classifier(checkpoint_path, auto_device=self.device)

            self.MLM, self.tokenizer = self.get_mlm_and_tokenizer(self.tad_classifier, _config)

            dataset_files = detect_dataset(dataset, task)
            boost_sets = dataset_files['valid']
            augmentations = []
            perplexity_list = []
            confidence_list = []

            for boost_set in boost_sets:
                print('Augmenting -> {}'.format(boost_set))
                fin = open(boost_set, encoding='utf8', mode='r')
                lines = fin.readlines()
                fin.close()
                remove(boost_set)
                for i in tqdm.tqdm(range(0, len(lines)), postfix='No.{} Augmenting...'.format(b_idx + 1)):

                    lines[i] = lines[i].strip().replace('$LABEL$', 'PLACEHOLDER')
                    label = lines[i].split('PLACEHOLDER')[1].strip()

                    if self.AUGMENT_BACKEND in 'EDA':
                        raw_augs = self.augmenter.augment(lines[i])
                    else:
                        raw_augs = self.augmenter.augment(lines[i], n=self.AUGMENT_NUM_PER_CASE, num_thread=os.cpu_count())

                    if isinstance(raw_augs, str):
                        raw_augs = [raw_augs]
                    augs = {}
                    for text in raw_augs:
                        if text.endswith('PLACEHOLDER {}'.format(label)) or text.endswith('PLACEHOLDER{}'.format(label)):
                            with torch.no_grad():
                                try:
                                    results = self.tad_classifier.infer(text.replace('PLACEHOLDER', '!ref!'), ignore_error=False, print_result=False)
                                except Exception as e:
                                    continue
                                ids = self.tokenizer(text, return_tensors="pt")
                                ids['labels'] = ids['input_ids'].clone()
                                ids = ids.to(self.device)
                                loss = self.MLM(**ids)['loss']
                                perplexity = torch.exp(loss / ids['input_ids'].size(1))

                                perplexity_list.append(perplexity.item())
                                confidence_list.append(results['confidence'])
                                if self.USE_LABEL:
                                    if results['ref_label_check'] != 'Correct':
                                        continue

                                if self.USE_CONFIDENCE:
                                    if results['confidence'] <= self.CONFIDENCE_THRESHOLD:
                                        continue

                                augs[perplexity.item()] = [text.replace('PLACEHOLDER', '$LABEL$')]

                    if self.USE_CONFIDENCE:
                        key_rank = sorted(augs.keys())
                    else:
                        key_rank = list(augs.keys())
                    for key in key_rank[:self.WINNER_NUM_PER_CASE]:
                        if self.USE_PERPLEXITY:
                            if key < self.PERPLEXITY_THRESHOLD:
                                augmentations += augs[key]
                        else:
                            augmentations += augs[key]

            print('Avg Confidence: {} Max Confidence: {} Min Confidence: {}'.format(np.average(confidence_list), max(confidence_list), min(confidence_list)))

            print('Avg Perplexity: {} Max Perplexity: {} Min Perplexity: {}'.format(np.average(perplexity_list), max(perplexity_list), min(perplexity_list)))

            fout = open('{}/{}.cross_boost.{}.train.augment.ignore'.format(os.path.dirname(dataset_file), fold_id, tag), encoding='utf8', mode='w')

            for line in augmentations:
                fout.write(line + '\n')
            fout.close()

            del self.tad_classifier
            del self.MLM

            torch.cuda.empty_cache()
            time.sleep(5)

            post_clean(os.path.dirname(dataset_file))

        for f in find_cwd_files('.augment.ignore'):
            rename(f, f.replace('.ignore', ''))

        if train_after_aug:
            print(colored('Start cross boosting augment...', 'green'))
            return Trainer(config=config,
                           dataset=dataset,  # train set and test set will be automatically detected
                           checkpoint_save_mode=0,  # =None to avoid save model
                           auto_device=self.device  # automatic choose CUDA or CPU
                           )

    def tad_mono_augment(self, config: ConfigManager,
                         dataset: DatasetItem,
                         rewrite_cache=True,
                         task='text_defense',
                         train_after_aug=False
                         ):
        if not isinstance(dataset, DatasetItem):
            dataset = DatasetItem(dataset)
        _config = self.get_tad_config(config)
        tag = '{}_{}_{}'.format(_config.model.__name__.lower(), dataset.dataset_name, os.path.basename(_config.pretrained_bert))

        prepare_dataset_and_clean_env(dataset.dataset_name, task, rewrite_cache)

        if not os.path.exists('checkpoints/mono_boost/{}'.format(tag)):
            os.makedirs('checkpoints/mono_boost/{}'.format(tag))

        print(colored('Begin mono boosting... ', 'yellow'))
        if self.WINNER_NUM_PER_CASE:

            keys = ['checkpoint', 'mono_boost', dataset.dataset_name, 'deberta']

            if len(find_dirs(self.ROOT, keys)) < self.CLASSIFIER_TRAINING_NUM + 1:
                # _config.log_step = -1
                Trainer(config=_config,
                        dataset=dataset,  # train set and test set will be automatically detected
                        checkpoint_save_mode=1,
                        path_to_save='checkpoints/mono_boost/{}/'.format(tag),
                        auto_device=self.device  # automatic choose CUDA or CPU
                        )

            torch.cuda.empty_cache()
            time.sleep(5)

            checkpoint_path = ''
            max_f1 = ''
            for path in find_dirs(self.ROOT, keys):
                if 'f1' in path and path[path.index('f1'):] > max_f1:
                    max_f1 = max(path[path.index('f1'):], checkpoint_path)
                    checkpoint_path = path

            self.tad_classifier = TADCheckpointManager.get_tad_text_classifier(checkpoint_path, auto_device=self.device)

            self.tad_classifier.opt.eval_batch_size = 128

            self.MLM, self.tokenizer = self.get_mlm_and_tokenizer(self.tad_classifier, _config)

            dataset_files = detect_dataset(dataset, task)
            boost_sets = dataset_files['train']
            augmentations = []
            perplexity_list = []
            confidence_list = []

            for boost_set in boost_sets:
                print('Augmenting -> {}'.format(boost_set))
                fin = open(boost_set, encoding='utf8', mode='r')
                lines = fin.readlines()
                fin.close()
                # remove(boost_set)
                for i in tqdm.tqdm(range(0, len(lines)), postfix='Mono Augmenting...'):

                    lines[i] = lines[i].strip().replace('$LABEL$', 'PLACEHOLDER')
                    label = lines[i].split('PLACEHOLDER')[1].strip()

                    if self.AUGMENT_BACKEND in 'EDA':
                        raw_augs = self.augmenter.augment(lines[i])
                    else:
                        raw_augs = self.augmenter.augment(lines[i], n=self.AUGMENT_NUM_PER_CASE, num_thread=os.cpu_count())

                    if isinstance(raw_augs, str):
                        raw_augs = [raw_augs]
                    augs = {}
                    for text in raw_augs:
                        if text.endswith('PLACEHOLDER {}'.format(label)) or text.endswith('PLACEHOLDER{}'.format(label)):
                            with torch.no_grad():
                                try:
                                    results = self.tad_classifier.infer(text.replace('PLACEHOLDER', '!ref!'), attack_defense=False, print_result=False)
                                except:
                                    continue
                                ids = self.tokenizer(text, return_tensors="pt")
                                ids['labels'] = ids['input_ids'].clone()
                                ids = ids.to(self.device)
                                loss = self.MLM(**ids)['loss']
                                perplexity = torch.exp(loss / ids['input_ids'].size(1))

                                perplexity_list.append(perplexity.item())
                                confidence_list.append(results['confidence'])

                                if results['ref_label_check'] == 'Correct' and results['confidence'] > self.CONFIDENCE_THRESHOLD:
                                    augs[perplexity.item()] = [text.replace('PLACEHOLDER', '$LABEL$')]

                    key_rank = sorted(augs.keys())
                    for key in key_rank[:self.WINNER_NUM_PER_CASE]:
                        if key < self.PERPLEXITY_THRESHOLD:
                            augmentations += augs[key]

            print('Avg Confidence: {} Max Confidence: {} Min Confidence: {}'.format(np.average(confidence_list), max(confidence_list), min(confidence_list)))

            print('Avg Perplexity: {} Max Perplexity: {} Min Perplexity: {}'.format(np.average(perplexity_list), max(perplexity_list), min(perplexity_list)))

            fout = open('{}/{}.mono_boost.train.augment.ignore'.format(os.path.dirname(boost_set), tag), encoding='utf8', mode='w')

            for line in augmentations:
                fout.write(line + '\n')
            fout.close()

            del self.tad_classifier
            del self.MLM

            torch.cuda.empty_cache()
            time.sleep(5)

            post_clean(os.path.dirname(boost_set))

        for f in find_cwd_files('.augment.ignore'):
            rename(f, f.replace('.augment.ignore', ''))

        if train_after_aug:
            print(colored('Start mono boosting augment...', 'yellow'))
            return Trainer(config=config,
                           dataset=dataset,  # train set and test set will be automatically detected
                           checkpoint_save_mode=0,  # =None to avoid save model
                           auto_device=self.device  # automatic choose CUDA or CPU
                           )


def query_dataset_detail(dataset_name, task='text_classification'):
    dataset_files = detect_dataset(dataset_name, task)
    data_dict = {}
    data_sum = 0
    if task in 'text_classification':
        for train_file in dataset_files['train']:
            with open(train_file, mode='r', encoding='utf8') as fin:
                lines = fin.readlines()
                for i in range(0, len(lines), 0):
                    data_dict[lines[i].strip()] = data_dict.get(lines[i].split('$LABEL$')[-1].strip(), 0) + 1
                    data_sum += 1
    else:
        for train_file in dataset_files['train']:
            with open(train_file, mode='r', encoding='utf8') as fin:
                lines = fin.readlines()
                for i in range(0, len(lines), 3):
                    data_dict[lines[i + 2].strip()] = data_dict.get(lines[i + 2].strip(), 0) + 1
                    data_sum += 1

    for label in data_dict:
        data_dict[label] = 1 - (data_dict[label] / data_sum)
    return data_dict


def post_clean(dataset_path):
    # if os.path.exists('{}/train.dat.tmp'.format(dataset_path)):
    #     remove('{}/train.dat.tmp'.format(dataset_path))
    # if os.path.exists('{}/valid.dat.tmp'.format(dataset_path)):
    #     remove('{}/valid.dat.tmp'.format(dataset_path))
    for f in find_files(dataset_path, '.tmp'):
        remove(f)
        remove(f + '.ignore')

    # for f in find_files(dataset_path, '.tmp.ignore', exclude_key='.augment.ignore'):
    #     remove(f)

    if find_cwd_dir('run'):
        shutil.rmtree(find_cwd_dir('run'))


def prepare_dataset_and_clean_env(dataset, task, rewrite_cache=False):
    # # download from local ABSADatasets
    if os.path.exists('integrated_datasets') and not os.path.exists('source_datasets.backup'):
        os.rename('integrated_datasets', 'source_datasets.backup')

    backup_datasets_dir = find_dir('source_datasets.backup', key=[dataset, task], disable_alert=True, recursive=True)

    datasets_dir = backup_datasets_dir.replace('source_datasets.backup', 'integrated_datasets')
    if not os.path.exists(datasets_dir):
        os.makedirs(datasets_dir)

    if rewrite_cache:
        print('Remove temp files (if any)')
        for f in find_files(datasets_dir, ['.augment']) + find_files(datasets_dir, ['.tmp']) + find_files(datasets_dir, ['.ignore']):
            # for f in find_files(datasets_dir, ['.tmp']):
            remove(f)
        os.system('rm {}/valid.dat.tmp'.format(datasets_dir))
        os.system('rm {}/train.dat.tmp'.format(datasets_dir))
        if find_cwd_dir(['run', dataset]):
            shutil.rmtree(find_cwd_dir(['run', dataset]))

        print('Remove Done')

    for f in os.listdir(backup_datasets_dir):
        if os.path.isfile(os.path.join(backup_datasets_dir, f)):
            shutil.copyfile(os.path.join(backup_datasets_dir, f), os.path.join(datasets_dir, f))
        elif os.path.isdir(os.path.join(backup_datasets_dir, f)):
            shutil.copytree(os.path.join(backup_datasets_dir, f), os.path.join(datasets_dir, f))


filter_key_words = ['.py', '.ignore', '.md', 'readme', 'log', 'result', 'zip', '.state_dict', '.model', '.png', 'acc_', 'f1_', '.aug', '.backup', '.bak']


def detect_dataset(dataset_path, task='apc'):
    if not isinstance(dataset_path, DatasetItem):
        dataset_path = DatasetItem(dataset_path)
    dataset_file = {'train': [], 'test': [], 'valid': []}
    for d in dataset_path:
        if not os.path.exists(d) or hasattr(ABSADatasetList, d) or hasattr(TCDatasetList, d):
            print('Loading {} dataset'.format(d))
            search_path = find_dir(os.getcwd(), ['integrated_datasets', d, task, 'dataset'], exclude_key=['infer', 'test.'] + filter_key_words, disable_alert=False)
            dataset_file['train'] += find_files(search_path, ['integrated_datasets', d, 'train', task], exclude_key=['.inference', 'test.'] + filter_key_words)
            dataset_file['test'] += find_files(search_path, ['integrated_datasets', d, 'test', task], exclude_key=['inference', 'train.'] + filter_key_words)
            dataset_file['valid'] += find_files(search_path, ['integrated_datasets', d, 'valid', task], exclude_key=['inference', 'train.'] + filter_key_words)
            dataset_file['valid'] += find_files(search_path, ['integrated_datasets', d, 'dev', task], exclude_key=['inference', 'train.'] + filter_key_words)
        else:
            dataset_file['train'] = find_files(d, ['integrated_datasets', 'train', task], exclude_key=['.inference', 'test.'] + filter_key_words)
            dataset_file['test'] = find_files(d, ['integrated_datasets', 'test', task], exclude_key=['.inference', 'train.'] + filter_key_words)
            dataset_file['valid'] = find_files(d, ['integrated_datasets', 'valid', task], exclude_key=['.inference', 'train.'] + filter_key_words)
            dataset_file['valid'] += find_files(d, ['integrated_datasets', 'dev', task], exclude_key=['inference', 'train.'] + filter_key_words)

    return dataset_file
