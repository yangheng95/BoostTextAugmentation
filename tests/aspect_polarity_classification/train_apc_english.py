# -*- coding: utf-8 -*-
# file: train_apc_english.py
# time: 2021/6/8 0008
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

########################################################################################################################
#                    train and evaluate on your own apc_datasets (need train and test apc_datasets)                    #
########################################################################################################################
import os
import warnings

import findfile

from pyabsa.functional import Trainer
from pyabsa.functional import APCConfigManager
from pyabsa.functional import ABSADatasetList
from pyabsa.functional import APCModelList

from transformers import AutoTokenizer, AutoModel

warnings.filterwarnings('ignore')

apc_config_english = APCConfigManager.get_apc_config_english()
apc_config_english.model = APCModelList.FAST_LCF_BERT
apc_config_english.num_epoch = 25
apc_config_english.patience = 10
apc_config_english.evaluate_begin = 0
apc_config_english.hidden_dim = 1024
apc_config_english.embed_dim = 1024
apc_config_english.evaluate_begin = 0
# apc_config_english.pretrained_bert = 'microsoft/deberta-v3-base'
apc_config_english.pretrained_bert = 'microsoft/deberta-v3-large'
# apc_config_english.pretrained_bert = './checkpoints/fast_lcf_bert_English_acc_83.41_f1_82.94/fine-tuned-pretrained-model/'
# apc_config_english.pretrained_bert = 'yangheng/deberta-v3-base-absa'
apc_config_english.similarity_threshold = 1
apc_config_english.max_seq_len = 80
apc_config_english.batch_size = 16
apc_config_english.dropout = 0
apc_config_english.seed = {7, 21, 53}
apc_config_english.log_step = 20
apc_config_english.l2reg = 1e-8
apc_config_english.learning_rate = 1e-5

for f in findfile.find_cwd_files('.augment.ignore'):
    os.rename(f, f.replace('.augment.ignore', '.augment'))

Dataset = ABSADatasetList.English
sent_classifier = Trainer(config=apc_config_english,
                          dataset=Dataset,
                          checkpoint_save_mode=3,
                          auto_device=True
                          ).load_trained_model()

examples = [
    'Strong build though which really adds to its [ASP]durability[ASP] .',  # !sent! Positive
    'Strong [ASP]build[ASP] though which really adds to its durability . !sent! Positive',
    'The [ASP]battery life[ASP] is excellent - 6-7 hours without charging . !sent! Positive',
    'I have had my computer for 2 weeks already and it [ASP]works[ASP] perfectly . !sent! Positive',
    'And I may be the only one but I am really liking [ASP]Windows 8[ASP] . !sent! Positive',
]

inference_sets = examples

for ex in examples:
    result = sent_classifier.infer(ex, print_result=True)
