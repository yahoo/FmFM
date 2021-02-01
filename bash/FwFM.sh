#!/usr/bin/env bash
model_type="FwFM"
model_folder=${data_folder}_${model_type}

python3.6 train.py \
--model_dir models/${model_folder} \
--train_data_file data/${data_folder}/train.csv \
--val_data_file data/${data_folder}/val.csv \
--test_data_file data/${data_folder}/test.csv \
--batch_size 1024 \
--train_epoch 20 \
--max_steps 50000 \
--l2_linear 1e-5 \
--l2_latent 1e-5 \
--l2_r 1e-5 \
--learning_rate 1e-4 \
--default_feat_dim 16 \
--feature_meta data/${data_folder}/features.json \
--feature_dict data/${data_folder}/feature_index \
--model_type $model_type

