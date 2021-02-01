# Copyright Verizon Media
# This project is licensed under the MIT. See license in project root for terms.

import tensorflow as tf
import logging
import json

logger = logging.getLogger(__name__)

def build_feature_meta(features_meta_file = 'features.json', features_dict_file = 'dict.txt'):
    ft_cat_counts = {}
    ft_types = {}
    ft_dims = {}
    ft_names = ['label']
    ft_defaults = [[0]]
    with tf.gfile.Open(features_meta_file, 'r') as ftfile:
        feature_meta = json.load(ftfile)
        for ft_name, ft_type, ft_dim in feature_meta:
            ft_names.append(ft_name)
            ft_types[ft_name] = ft_type
            ft_dims[ft_name] = ft_dim
            if ft_type == 'CATEGORICAL':
                ft_defaults.append([0])
            elif ft_type == 'NUMERIC':
                ft_defaults.append([0.0])

    with tf.gfile.Open(features_dict_file) as fm_file:
        logger.info('Reading features from file %s', features_dict_file)
        for feature in fm_file:
            ft = feature.strip().split('\1')
            feature_name = ft[0].strip()
            if ft_cat_counts.get(feature_name) is None:
                ft_cat_counts[feature_name] = 1
            else:
                ft_cat_counts[feature_name] += 1

    return ft_names, ft_defaults, ft_cat_counts, ft_types, ft_dims


def load_cross_fields(cross_fields_file):
    if cross_fields_file is None:
      return None
    else:
      return set(json.load(open(cross_fields_file)))


# Create a feature
def parse_record(record, feature_names, feature_defaults):
    feature_array = tf.decode_csv(record, feature_defaults)
    features = dict(zip(feature_names, feature_array))
    label = features.pop('label')
    #features.pop('tag') # unused
    # if features['ads_category'] < 0:
    #     features['ads_category'] = 0
    return features, label


def input_fn(train_files, shuffle, batch_size, epoch, feature_names, feature_defaults):
    dataset = tf.data.TextLineDataset(train_files)
    if epoch:
        dataset = dataset.repeat(epoch)

    if shuffle:
        dataset = dataset.shuffle(shuffle)

    dataset = dataset.map(lambda x: parse_record(x, feature_names, feature_defaults))
    dataset = dataset.batch(batch_size)

    return dataset


if __name__ == "__main__":
    feature_names, feature_defaults, categorical_feature_counts, feature_types, feature_dim = \
        build_feature_meta('data/criteo/feature.json', 'data/criteo/feature_index')

    print(feature_dim)