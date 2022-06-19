# -*- coding: utf-8 -*-
# file: augment_test.py
# time: 07/06/2022
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.
import os

import autocuda
from pyabsa import TCConfigManager, GloVeTCModelList, TCDatasetList, BERTTCModelList

from boost_aug import TCBoostAug, AugmentBackend

device = autocuda.auto_cuda()

tc_config = TCConfigManager.get_classification_config_english()
tc_config.model = BERTTCModelList.BERT  # 'BERT' model can be used for DeBERTa or BERT
tc_config.num_epoch = 15
tc_config.evaluate_begin = 0
tc_config.max_seq_len = 100
tc_config.pretrained_bert = 'microsoft/deberta-v3-base'
tc_config.log_step = 100
tc_config.dropout = 0.1
tc_config.cache_dataset = False
tc_config.seed = 1
tc_config.l2reg = 1e-7
tc_config.learning_rate = 1e-5

backend = AugmentBackend.EDA
dataset = TCDatasetList.SST2

BoostingAugmenter = TCBoostAug(ROOT=os.getcwd(),
                               AUGMENT_BACKEND=backend,
                               WINNER_NUM_PER_CASE=8,
                               AUGMENT_NUM_PER_CASE=16,
                               PERPLEXITY_THRESHOLD=3,
                               device=device)

augs = BoostingAugmenter.single_augment('culkin exudes none of the charm or charisma that might keep a more general audience even vaguely interested in his bratty character .', 0, 3)
print(augs)
