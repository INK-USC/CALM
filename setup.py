#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import setuptools

with open("README.md") as f:
    readme = f.read()

setuptools.setup(
    name="calm",
    version="1.0.0",
    description="Pre-training Text-to-Text Transformers for Concept-centric Common Sense",
    url="https://github.com/INK-USC/CALM/",
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    long_description=readme,
    long_description_content_type="text/markdown",
    setup_requires=[
        "setuptools>=18.0",
    ],
    package_dir={"": "CALM"},
    packages=setuptools.find_packages(where="CALM"),
    python_requires=">=3.6",
    install_requires=[
        "nltk == 3.5",
        "numpy == 1.19.4",
        "pandas == 1.1.4",
        "pytorch-lightning == 0.9.0",
        "scikit-learn == 0.23.2",
        "sentencepiece == 0.1.94",
        "spacy == 2.3.4",
        "tensorboard == 2.2.0",
        "torch == 1.7.0",
        "transformers == 4.1.1"
    ],
)
