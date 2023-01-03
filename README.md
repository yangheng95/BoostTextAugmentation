# Boosting Data Augmentation for Text Classification

[![Downloads](https://pepy.tech/badge/boostaug)](https://pepy.tech/project/boostaug)
[![Downloads](https://pepy.tech/badge/boostaug/month)](https://pepy.tech/project/boostaug)
[![Downloads](https://pepy.tech/badge/boostaug/week)](https://pepy.tech/project/boostaug)


# Usage in PyABSA
you can find examples for augmenting text classification and aspect-term sentiment classification at https://github.com/yangheng95/PyABSA/tree/v2/examples-v2/augmentation

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

## MUST READ

- If the augmentation traning is terminated by accidently or you want to rerun augmentation, set `rewrite_cache=True`
  in augmentation.
- If you have many datasets, run augmentation for differnet datasets IN SEPARATE FOLDER, otherwise `IO OPERATION`
  may CORRUPT other datasets

# Notice

This is the draft code, so do not perform cross-boosting on different dataset in the same folder, which will raise some
Exception
