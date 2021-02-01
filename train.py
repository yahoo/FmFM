# Copyright Verizon Media
# This project is licensed under the MIT. See license in project root for terms.

from __future__ import (absolute_import, division, print_function)
import tensorflow as tf
from os.path import join
from os import getenv
import logging
import features
import models
import os

logger = logging.getLogger(__name__)

def report_update(file_ptr, mode, itr, metrics):
    info = 'Iter:%d\t%s Set metrics - %s\n' % (itr+1, mode, format(metrics))
    file_ptr.write(info)
    file_ptr.flush()
    return info

def main(args):
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    report_file = open(os.path.join(args.model_dir, 'report.txt'), 'w+')
    tf.logging.set_verbosity(tf.logging.INFO)
    logging.basicConfig(level=logging.INFO)

    logger.info('Training start with arguments... {}'.format(args))
    logger.info('tf version {}'.format(tf.__version__))
    logger.info('TF_CONFIG : {}'.format(getenv('TF_CONFIG')))

    logger.info('Starting to load feature info from {},{}'.format(args.feature_meta, args.feature_dict))
    feature_names, feature_defaults, categorical_feature_counts, feature_types, feature_dims = \
        features.build_feature_meta(args.feature_meta, args.feature_dict)

    cross_fields = features.load_cross_fields(args.cross_fields)

    print('{}'.format(feature_names))
    print('{}'.format(feature_defaults))
    print('{}'.format(categorical_feature_counts))
    print('{}'.format(sorted(feature_types.items())))

    logger.info('train file: {}'.format(args.train_data_file))
    logger.info('val file: {}'.format(args.val_data_file))
    logger.info('test file: {}'.format(args.test_data_file))

    def train_input_fn():
        return features.input_fn(args.train_data_file, True, args.batch_size, None,
                                 feature_names, feature_defaults)
    def val_input_fn():
        return features.input_fn(args.val_data_file, True, args.batch_size, None,
                                 feature_names, feature_defaults)
    def test_input_fn():
        return features.input_fn(args.test_data_file, True, args.batch_size, None,
                                 feature_names, feature_defaults)

    model = models.build_custom_linear_classifier(args.model_dir, feature_names, feature_types,
                    categorical_feature_counts, args.l2_linear, args.l2_latent, args.l2_r, args.learning_rate,
                    args.default_feat_dim, args.model_type, feature_dims, cross_fields)

    tf.reset_default_graph()
    best_auc = 0


    for n in range(args.train_epoch):
        model.train(input_fn=train_input_fn)

        metrics = model.evaluate(input_fn=val_input_fn, name='Val')
        logger.info(report_update(report_file, 'Val', n, metrics))
        if metrics['auc'] > best_auc:
            best_auc = metrics['auc']
        else:
            break

    metrics = model.evaluate(input_fn=test_input_fn, name='Test')
    logger.info(report_update(report_file, 'Test', n, metrics))

    metrics = model.evaluate(input_fn=train_input_fn, name='train')
    logger.info(report_update(report_file, 'Train', n, metrics))

    report_file.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_dir", help="path to save model/checkpoint", default="dsp_cpc_tf_model")
    parser.add_argument("--train_data_file", help="path to training input data", default="dsp_cpc_train")
    parser.add_argument("--val_data_file", help="path to validate input data", default="dsp_cpc_val")
    parser.add_argument("--test_data_file", help="path to test input data", default="dsp_cpc_test")
    parser.add_argument("--batch_size", help="number of records per batch", type=int, default=100)
    parser.add_argument("--eval_batch_size", help="number of records per batch for eval", type=int, default=100)
    parser.add_argument("--train_epoch", help="Training epoch count", type=int, default=20)
    parser.add_argument("--max_steps", help="maximum number of steps", type=int, default=100000)
    parser.add_argument("--l2_linear", help="l2 regularization scale for linear term", type=float, default=0.0)
    parser.add_argument("--l2_latent", help="l2 regularization scale for latent factor", type=float, default=0.0)
    parser.add_argument("--l2_r", help="l2 regularization scale for fields", type=float, default=0.0)
    parser.add_argument("--learning_rate", help="learning rate in Adam Optimizer", type=float, default=1e-4)
    parser.add_argument("--default_feat_dim", help="dimension of latent vector for each feature", type=int, default=10)
    parser.add_argument("--cross_fields", help="json file, list of cross fields", default=None)
    parser.add_argument("--model_type", help="the model used in training (lr, FM, FwFM)", default="LR")
    parser.add_argument("--feature_meta", help="path to feature meta data", default="features.json")
    parser.add_argument("--feature_dict", help="path to feature dictionary", default="feature_dict")


    args = parser.parse_args()
    print("args:", args)

    print('Executing in local mode')
    main(args)