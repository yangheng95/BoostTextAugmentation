# -*- coding: utf-8 -*-
# file: sst_augmentation.py
# time: 12/06/2022
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.
import os

import autocuda
import random
import warnings
from pyabsa.functional.dataset import DatasetItem

from pyabsa import TCDatasetList
from pyabsa.functional import GloVeTCModelList
from pyabsa.functional import TADConfigManager
from pyabsa.functional import TADBERTTCModelList

from boost_aug import TCBoostAug, AugmentBackend, TADBoostAug

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

for dataset in [
    # DatasetItem('SST2BAE'),
    # DatasetItem('SST2PWWS'),
    DatasetItem('SST2TextFooler'),
]:
    for backend in aug_backends:
        tad_config = TADConfigManager.get_tad_config_english()
        tad_config.model = TADBERTTCModelList.TADBERT  # 'BERT' model can be used for DeBERTa or BERT
        tad_config.num_epoch = 15
        tad_config.evaluate_begin = 0
        tad_config.max_seq_len = 100
        tad_config.log_step = -1
        tad_config.dropout = 0.1
        tad_config.cache_dataset = False
        tad_config.seed = seeds
        tad_config.l2reg = 1e-7
        tad_config.learning_rate = 1e-5

        BoostingAugmenter = TADBoostAug(ROOT=os.getcwd(),
                                        AUGMENT_BACKEND=backend,
                                        CLASSIFIER_TRAINING_NUM=2,
                                        WINNER_NUM_PER_CASE=8,
                                        AUGMENT_NUM_PER_CASE=16,
                                        device=device)
        BoostingAugmenter.tad_boost_augment(tad_config,
                                            dataset,
                                            train_after_aug=True,
                                            rewrite_cache=True,
                                            )
