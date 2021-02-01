# FM^2: Field-matrixed Factorization Machines for Recommender Systems

## Table of Contents

- [Background](#background)
- [Install](#install)
- [Usage](#usage)
- [Contribute](#contribute)
- [License](#license)

## Background
This is the code to implement the algorithm of FM^2 (Field-matrixed Factorization Machines), it can run a quick benchmark among the LR, FM, FFM, FwFM, FvFM, FmFM and DCN,
it also support data process and feature extraction from public data set Criteo and Avazu.


## Install
First you will need to have [TensorFlow](https://github.com/tensorflow) (v1.15 with a GPU is preferred) and numpy, pandas, pickle and tqdm installed.

You may need to login and download the [Criteo](http://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/) and [Avazu](https://www.kaggle.com/c/avazu-ctr-prediction/data) from their websites respectively.
The unzipped raw data files should be placed at folder `data/criteo/` and `data/avazu/` respectively.

## Usage

This project has the following content
1. **train.py** The main function to train the model
2. **features.py** Functions to process the data file and generate features
3. **models.py** The core functions to describe those models, include the new proposed FmFM and FvFM, as well as other baseline models like LR, FM, FFM, FwFM

The folder **bash** contains individual training task with hyper-parameters, and the **start_train.sh** can schedule multiple task in one bash file.  

![AUC vs FLOP comparison](/auc_flop.png)

## Contribute

Please refer to [the contributing.md file](Contributing.md) for information about how to get involved. We welcome issues, questions, and pull requests.

## Maintainers
Yang Sun, yang.sun@verizonmedia.com

## License
This project is licensed under the terms of the MIT open source license. Please refer to LICENSE for the full terms.
