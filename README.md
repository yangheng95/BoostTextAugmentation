# Boosting Data Augmentation for Text Classification

[![Downloads](https://pepy.tech/badge/boostaug)](https://pepy.tech/project/boostaug)
[![Downloads](https://pepy.tech/badge/boostaug/month)](https://pepy.tech/project/boostaug)
[![Downloads](https://pepy.tech/badge/boostaug/week)](https://pepy.tech/project/boostaug)

Codes for ACL2023 Findings paper: [Boosting Text Augmentation via Hybrid Instance Filtering Framework](https://aclanthology.org/2023.findings-acl.105)


# Usage in PyABSA
you can find examples for augmenting text classification and aspect-term sentiment classification at https://github.com/yangheng95/PyABSA/tree/v2/examples-v2/text_augmentation

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

# Citation
```
@inproceedings{yang-li-2023-boosting,
    title = "Boosting Text Augmentation via Hybrid Instance Filtering Framework",
    author = "Yang, Heng  and
      Li, Ke",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-acl.105",
    pages = "1652--1669",
}
```
