# -*- coding: utf-8 -*-
# file: setup.py
# time: 2021/4/22 0022
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

from setuptools import setup, find_packages

from boost_aug import __version__, __name__

setup(
    name=__name__,
    version=__version__,
    description='',
    url='https://github.com/yangheng95/BoostAug',
    # Author details
    author='Yang Heng',
    author_email='hy345@exeter.ac.uk',
    python_requires=">=3.6",
    packages=find_packages(),
    include_package_data=True,
    exclude_package_date={'': ['.gitignore']},
    # Choose your license
    license='MIT',
    install_requires=[
        'pyabsa>=1.16.18',
        # install the following requirements depend on the backend of your choice
        # 'textattack',
        # 'nlpaug',
        # 'tensorflow_text'
    ],
)
