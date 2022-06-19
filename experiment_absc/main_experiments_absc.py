import os

import autocuda
import random
import warnings

from boost_aug import ABSCBoostAug, AugmentBackend

from pyabsa.functional import APCConfigManager
from pyabsa.functional import ABSADatasetList
from pyabsa.functional import APCModelList, GloVeAPCModelList

warnings.filterwarnings('ignore')

aug_backends = [
    AugmentBackend.EDA,
    # AugmentBackend.SplitAug,
    # AugmentBackend.SpellingAug,
    # AugmentBackend.ContextualWordEmbsAug,  # WordEmbsAug
    # AugmentBackend.BackTranslationAug,

]
device = autocuda.auto_cuda()

# seeds = [random.randint(0, 10000) for _ in range(5)]
seeds = [random.randint(0, 10000) for _ in range(1)]

for backend in aug_backends:
    for dataset in [
        ABSADatasetList.Laptop14,
        ABSADatasetList.Restaurant14,
        ABSADatasetList.Restaurant15,
        ABSADatasetList.Restaurant16,
        ABSADatasetList.MAMS
    ]:
        config = APCConfigManager.get_apc_config_english()
        config.model = APCModelList.FAST_LCF_BERT
        config.lcf = 'cdw'
        config.similarity_threshold = 1
        config.max_seq_len = 80
        config.dropout = 0
        config.optimizer = 'adam'
        config.cache_dataset = False
        config.pretrained_bert = 'microsoft/deberta-v3-base'
        config.hidden_dim = 768
        config.embed_dim = 768
        config.log_step = 20
        config.SRD = 3
        config.learning_rate = 1e-5
        config.batch_size = 16
        config.num_epoch = 30
        config.evaluate_begin = 2
        config.l2reg = 1e-8
        config.seed = seeds

        BoostingAugmenter = ABSCBoostAug(ROOT=os.getcwd(), AUGMENT_BACKEND=backend, AUGMENT_NUM_PER_CASE=16, WINNER_NUM_PER_CASE=8, device=device)
        BoostingAugmenter.apc_boost_augment(config,  # BOOSTAUG
                                            dataset,
                                            train_after_aug=True,
                                            rewrite_cache=True,
                                            )
        BoostingAugmenter.apc_classic_augment(config,  # prototype Aug
                                              dataset,
                                              train_after_aug=True,
                                              rewrite_cache=True,
                                              )
        BoostingAugmenter.apc_mono_augment(config,  # MonoAUG
                                           dataset,
                                           train_after_aug=True,
                                           rewrite_cache=True,
                                           )

for backend in aug_backends:
    for dataset in [
        ABSADatasetList.Laptop14,
        ABSADatasetList.Restaurant14,
        ABSADatasetList.Restaurant15,
        ABSADatasetList.Restaurant16,
        ABSADatasetList.MAMS
    ]:
        config = APCConfigManager.get_apc_config_english()
        config.model = APCModelList.BERT_SPC
        config.lcf = 'cdw'
        config.similarity_threshold = 1
        config.max_seq_len = 80
        config.dropout = 0
        config.optimizer = 'adam'
        config.cache_dataset = False
        config.pretrained_bert = 'bert-base-uncased'
        config.hidden_dim = 768
        config.embed_dim = 768
        config.log_step = 20
        config.SRD = 3
        config.learning_rate = 1e-5
        config.batch_size = 16
        config.num_epoch = 30
        config.evaluate_begin = 2
        config.l2reg = 1e-8
        config.seed = seeds

        BoostingAugmenter = ABSCBoostAug(ROOT=os.getcwd(), AUGMENT_BACKEND=backend, AUGMENT_NUM_PER_CASE=16, WINNER_NUM_PER_CASE=8, device=device)
        BoostingAugmenter.apc_boost_augment(config,
                                            dataset,
                                            train_after_aug=True,
                                            rewrite_cache=True,
                                            )
        BoostingAugmenter.apc_classic_augment(config,
                                              dataset,
                                              train_after_aug=True,
                                              rewrite_cache=True,
                                              )
        BoostingAugmenter.apc_mono_augment(config,
                                           dataset,
                                           train_after_aug=True,
                                           rewrite_cache=True,
                                           )

for backend in aug_backends:
    for dataset in [
        ABSADatasetList.Laptop14,
        ABSADatasetList.Restaurant14,
        ABSADatasetList.Restaurant15,
        ABSADatasetList.Restaurant16,
        ABSADatasetList.MAMS
    ]:
        apc_config = APCConfigManager.get_apc_config_glove()
        apc_config.model = GloVeAPCModelList.LSTM
        apc_config.max_seq_len = 100
        apc_config.dropout = 0
        apc_config.optimizer = 'adam'
        apc_config.cache_dataset = False
        apc_config.learning_rate = 0.001
        apc_config.batch_size = 64
        apc_config.num_epoch = 100
        apc_config.evaluate_begin = 0
        apc_config.l2reg = 1e-4
        apc_config.log_step = 5
        apc_config.seed = seeds

        BoostingAugmenter = ABSCBoostAug(ROOT=os.getcwd(), AUGMENT_BACKEND=backend, AUGMENT_NUM_PER_CASE=16, WINNER_NUM_PER_CASE=8, device=device)
        BoostingAugmenter.apc_boost_augment(config,
                                            dataset,
                                            train_after_aug=True,
                                            rewrite_cache=True,
                                            )
        BoostingAugmenter.apc_classic_augment(config,
                                              dataset,
                                              train_after_aug=True,
                                              rewrite_cache=True,
                                              )
        BoostingAugmenter.apc_mono_augment(config,
                                           dataset,
                                           train_after_aug=True,
                                           rewrite_cache=True,
                                           )
