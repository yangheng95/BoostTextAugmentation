# Boosting Augment Framework for Text Classification

## Install BoostAug

### Install from pip

```bash
pip install BoostAug
```

### Install from source

```bash
git clone https://github.com/yangheng95/BoostingAugABSA
cd BoostingAugABSA
pip install .
```

## Quick Start

We made a package BoostAug which can helps while using our code, here are the examples for using BoostAug to improve
aspect-level polarity classification and sentence-level text classification,

### Import BoostAug

```python3
import shutil
import autocuda
import findfile
import random
import warnings

from boost_aug import BoostingAug, AugmentBackend

from pyabsa.functional import APCConfigManager
from pyabsa.functional import ABSADatasetList
from pyabsa.functional import APCModelList

warnings.filterwarnings('ignore')
```

### Create a BoostingAugmenter
```python3
aug_backend = AugmentBackend.EDA

# BoostingAugmenter = BoostingAug(AUGMENT_BACKEND=aug_backend, device='cuda')
BoostingAugmenter = BoostingAug(AUGMENT_BACKEND=aug_backend, device=autocuda.auto_cuda())

```

### Boosting Augment for APC

```python3
BoostingAugmenter = BoostingAug()
device = autocuda.auto_cuda()

seeds = [random.randint(0, 10000) for _ in range(5)]

apc_config_english = APCConfigManager.get_apc_config_english()
apc_config_english.model = APCModelList.FAST_LCF_BERT
apc_config_english.cache_dataset = False
apc_config_english.patience = 10
apc_config_english.pretrained_bert = 'microsoft/deberta-v3-base'
apc_config_english.log_step = 50
apc_config_english.learning_rate = 1e-5
apc_config_english.num_epoch = 25
apc_config_english.l2reg = 1e-8
apc_config_english.seed = seeds
apc_config_english.cross_validate_fold = -1  # disable cross_validate

BoostingAugmenter.apc_boost_free_training(apc_config_english,
                                          ABSADatasetList.Laptop14)

BoostingAugmenter.apc_cross_boost_training(apc_config_english,
                                           ABSADatasetList.Laptop14,
                                           rewrite_cache=True)

BoostingAugmenter.apc_classic_boost_training(apc_config_english,
                                             ABSADatasetList.Laptop14)

```

### Boosting Augment for Text Classification

```python3

seeds = [random.randint(0, 10000) for _ in range(5)]

apc_config_english = APCConfigManager.get_apc_config_english()
apc_config_english.model = APCModelList.FAST_LCF_BERT
apc_config_english.cache_dataset = False
apc_config_english.patience = 10
apc_config_english.pretrained_bert = 'microsoft/deberta-v3-base'
apc_config_english.log_step = 50
apc_config_english.learning_rate = 1e-5
apc_config_english.num_epoch = 25
apc_config_english.l2reg = 1e-8
apc_config_english.seed = seeds
apc_config_english.cross_validate_fold = -1  # disable cross_validate

BoostingAugmenter.tc_boost_free_training(apc_config_english,
                                         ABSADatasetList.Laptop14)

BoostingAugmenter.tc_cross_boost_training(apc_config_english,
                                          ABSADatasetList.Laptop14,
                                          rewrite_cache=True)

BoostingAugmenter.tc_classic_boost_training(apc_config_english,
                                             ABSADatasetList.Laptop14)
```
