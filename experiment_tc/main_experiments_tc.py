import os

import autocuda
import random
import warnings

from pyabsa import TCDatasetList
from pyabsa.functional import GloVeTCModelList
from pyabsa.functional import TCConfigManager
from pyabsa.functional import BERTTCModelList

from boost_aug import TCBoostAug, AugmentBackend

warnings.filterwarnings('ignore')

device = autocuda.auto_cuda()
# seeds = [random.randint(0, 10000) for _ in range(5)]  # experiment
seeds = [random.randint(0, 10000) for _ in range(1)]  # TEST

aug_backends = [
    AugmentBackend.EDA,
    # AugmentBackend.SplitAug,
    # AugmentBackend.SpellingAug,
    # AugmentBackend.ContextualWordEmbsAug,
    # AugmentBackend.BackTranslationAug,
]

# -----------------------------------------------------------------------------#
#                              GloVe Experiments                               #
#       Please download glove.840B.300d.txt in current working directory       #
#                         before run glove experiments                         #
# -----------------------------------------------------------------------------#
# for dataset in [
#     TCDatasetList.SST2,
#     # TCDatasetList.SST5,
#     TCDatasetList.AGNews10K,
#     TCDatasetList.Yelp10K
# ]:
#     for backend in aug_backends:
#         tc_config = TCConfigManager.get_classification_config_glove()
#         tc_config.model = GloVeTCModelList.LSTM
#         tc_config.max_seq_len = 100
#         tc_config.dropout = 0
#         tc_config.optimizer = 'adam'
#         tc_config.cache_dataset = False
#         tc_config.patience = 20
#         tc_config.learning_rate = 0.001
#         tc_config.batch_size = 128
#         tc_config.num_epoch = 150
#         tc_config.evaluate_begin = 0
#         tc_config.l2reg = 1e-4
#         tc_config.log_step = 5
#         tc_config.seed = seeds
#         tc_config.cross_validate_fold = -1  # disable cross_validate
#
#         BoostingAugmenter = TCBoostAug(ROOT=os.getcwd(),
#                                        AUGMENT_BACKEND=backend,
#                                        CLASSIFIER_TRAINING_NUM=1,
#                                        WINNER_NUM_PER_CASE=8,
#                                        AUGMENT_NUM_PER_CASE=16,
#                                        device=device)
#         BoostingAugmenter.tc_boost_augment(tc_config,  # BOOSTAUG
#                                            dataset,
#                                            train_after_aug=True,
#                                            rewrite_cache=True,
#                                            )
#         # BoostingAugmenter.tc_classic_boost_training(tc_config,               # prototype Aug
#         #                                            dataset,
#         #                                            train_after_aug=True,
#         #                                            rewrite_cache=True,
#         #                                            )
#         # BoostingAugmenter.tc_mono_boost_training(tc_config,                  # MonoAUG
#         #                                            dataset,
#         #                                            train_after_aug=True,
#         #                                            rewrite_cache=True,
#         #                                            )
#         # BoostingAugmenter.tc_boost_free_training(tc_config,                  # Non-Aug
#         #                                            dataset
#         #                                            )

# -----------------------------------------------------------------------------#
#                             DeBERTa Experiments                              #
for dataset in [
    # TCDatasetList.SST2,
    # TCDatasetList.SST5,
    # TCDatasetList.AGNews10K,
    TCDatasetList.IMDB10K,
    # TCDatasetList.Yelp10K
]:
    for backend in aug_backends:
        tc_config = TCConfigManager.get_classification_config_english()
        tc_config.model = BERTTCModelList.BERT  # 'BERT' model can be used for DeBERTa or BERT
        tc_config.num_epoch = 15
        tc_config.evaluate_begin = 0
        tc_config.max_seq_len = 512
        tc_config.pretrained_bert = 'microsoft/deberta-v3-base'
        tc_config.log_step = -1
        tc_config.dropout = 0.1
        tc_config.cache_dataset = False
        tc_config.seed = seeds
        tc_config.l2reg = 1e-7
        tc_config.learning_rate = 1e-5

        BoostingAugmenter = TCBoostAug(ROOT=os.getcwd(),
                                       AUGMENT_BACKEND=backend,
                                       CLASSIFIER_TRAINING_NUM=2,
                                       WINNER_NUM_PER_CASE=8,
                                       AUGMENT_NUM_PER_CASE=16,
                                       device=device)
        # BoostingAugmenter.tc_boost_augment(tc_config,
        #                                    dataset,
        #                                    train_after_aug=True,
        #                                    rewrite_cache=True,
        #                                    )
        # BoostingAugmenter.tc_classic_boost_training(tc_config,
        #                                            dataset,
        #                                            train_after_aug=True,
        #                                            rewrite_cache=True,
        #                                            )
        BoostingAugmenter.tc_mono_augment(tc_config,
                                                   dataset,
                                                   train_after_aug=True,
                                                   rewrite_cache=True,
                                                   )
        # BoostingAugmenter.tc_boost_free_training(tc_config,
        #                                            dataset
        #                                            )

# -----------------------------------------------------------------------------#
#                               BERT Experiments                               #
for dataset in [
    TCDatasetList.SST2,
    TCDatasetList.SST5,
    TCDatasetList.AGNews10K,
]:
    for backend in aug_backends:
        tc_config = TCConfigManager.get_classification_config_english()
        tc_config.model = BERTTCModelList.BERT  # 'BERT' model can be used for DeBERTa or BERT
        tc_config.num_epoch = 15
        tc_config.evaluate_begin = 0
        tc_config.max_seq_len = 100
        tc_config.pretrained_bert = 'bert-base-uncased'
        tc_config.log_step = 100
        tc_config.dropout = 0.1
        tc_config.cache_dataset = False
        tc_config.seed = seeds
        tc_config.l2reg = 1e-7
        tc_config.learning_rate = 1e-5

        BoostingAugmenter = TCBoostAug(ROOT=os.getcwd(),
                                       AUGMENT_BACKEND=backend,
                                       WINNER_NUM_PER_CASE=8,
                                       AUGMENT_NUM_PER_CASE=16,
                                       device=device)
        BoostingAugmenter.tc_boost_augment(tc_config,
                                           dataset,
                                           train_after_aug=True,
                                           rewrite_cache=True,
                                           )
        # BoostingAugmenter.tc_classic_boost_training(tc_config,
        #                                            dataset,
        #                                            train_after_aug=True,
        #                                            rewrite_cache=True,
        #                                            )
        # BoostingAugmenter.tc_mono_boost_training(tc_config,
        #                                            dataset,
        #                                            train_after_aug=True,
        #                                            rewrite_cache=True,
        #                                            )
        # BoostingAugmenter.tc_boost_free_training(tc_config,
        #                                            dataset
        #                                            )
