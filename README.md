# Boosting Data Augmentation for Text Classification

[![Downloads](https://pepy.tech/badge/boostaug)](https://pepy.tech/project/boostaug)
[![Downloads](https://pepy.tech/badge/boostaug/month)](https://pepy.tech/project/boostaug)
[![Downloads](https://pepy.tech/badge/boostaug/week)](https://pepy.tech/project/boostaug)

## Notice

This tool depends on the [PyABSA](https://github.com/yangheng95/PyABSA),
and is integrated with the [ABSADatasets](https://github.com/yangheng95/ABSADatasets).

To augment your own dataset, you need to prepare your dataset according to **ABSADatasets**.
Refer to the [instruction to process](https://github.com/yangheng95/ABSADatasets)
or [annotate your dataset](https://github.com/yangheng95/ABSADatasets/tree/v1.2/DPT).

## Install BoostAug

### Install from source

```bash
git clone https://github.com/yangheng95/BoostTextAugmentation

cd BoostTextAugmentation

pip install .
```

## Quick Start

We made a package which can helps while using our code, here are the examples for using augmentation to improve
aspect-level polarity classification and sentence-level text classification

If there is no enough resource to run augmentation, we have done some augmentation and
prepared some augmentation sets in the dataset folders, please set `rewrite_cache=False` to run training on these
augmentation sets.
e.g.,

### Usage

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

### Augmentation and Training

```python3
BoostingAugmenter.apc_boost_augment(config,  
                                    dataset,
                                    train_after_aug=True,  # comment this line to perform augmentation without training
                                    rewrite_cache=False, # use pre-augmented datasets for training to evaluate performance
                                    )

```

## MUST READ

- If the augmentation traning is terminated by accidently or you want to rerun augmentation, set `rewrite_cache=True`
  in augmentation.
- If you have many datasets, run augmentation for differnet datasets IN SEPARATE FOLDER, otherwise `IO OPERATION`
  may CORRUPT other datasets

# Notice

This is the draft code, so do not perform cross-boosting on different dataset in the same folder, which will raise some
Exception
