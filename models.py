# Copyright Verizon Media
# This project is licensed under the MIT. See license in project root for terms.

import tensorflow as tf
import numpy as np
import pandas as pd
from os.path import join
import logging
import pickle


logger = logging.getLogger(__name__)

def write_weight(outfile, weight, rec):
    if weight != 0.0:
        outfile.write(rec)


def build_feature_columns(feature_names, feature_types, categorical_feature_counts):
    features = []
    for feature_name in feature_names:
        if feature_name == 'label' or feature_name == 'tag':
            pass
        elif feature_types[feature_name] == 'CATEGORICAL':
            f1 = tf.feature_column.categorical_column_with_identity(feature_name, categorical_feature_counts[feature_name], default_value=0)
            features.append(f1)
        elif feature_types[feature_name] == 'NUMERIC':
            features.append(tf.feature_column.numeric_column(feature_name, default_value=0.0))
    return features


def metric_auc(labels, predictions):
    return {
        'auc_precision_recall': tf.metrics.auc(
            labels=labels, predictions=predictions['logistic'], num_thresholds=200,
            curve='PR', summation_method='careful_interpolation')
    }


def export_model(model, model_export_path, feature_names, feature_types, categorical_feature_counts):
    return model.export_savedmodel(model_export_path,
            tf.estimator.export.build_parsing_serving_input_receiver_fn(
                tf.feature_column.make_parse_example_spec(build_feature_columns(
                    feature_names, feature_types, categorical_feature_counts
                )),
            ), as_text=True)


def save_model_weights(model, model_weights_dir, feature_types):
    if not tf.gfile.Exists(model_weights_dir):
        tf.gfile.MkDir(model_weights_dir)

    print('Saving model weights to {}'.format(model_weights_dir))
    with tf.gfile.Open(join(model_weights_dir, 'model.txt'), 'w+') as modelfile:
        dense_weight = model.get_variable_value('dense/kernel')[0][1]
        bias = model.get_variable_value('linear_model/bias_weights') * dense_weight + model.get_variable_value('dense/bias')[1]
        # format is feature_column_name, feature_idx, weights
        write_weight(modelfile, bias, '{},{},{}\n'.format('BiasWeight', 0, bias))
        for f_order, (feature_column, feature_type) in enumerate(feature_types.items()):
            if feature_type == 'CATEGORICAL':
                t = model.get_variable_value('linear_model/' + feature_column + '/weights')
                for ft_idx, wt in enumerate(t):
                    write_weight(modelfile, wt[0] * dense_weight, '{},{},{}\n'.format(feature_column, ft_idx, wt[0] * dense_weight))
            else:
                write_weight(modelfile, dense_weight, '{},{},{}\n'.format(feature_column, 0, dense_weight))


def generate_field_dic(path, field):
    '''read and create the field feature index - value dictionary'''
    df = pd.read_csv(path, sep=chr(1), header=None)
    df.columns = ['field', 'value', 'index']
    df_line = df[df['field'] == field]
    field_dic = {}
    for index, row in df_line.iterrows():
        # access data using column names
        field_dic[row['index']] = row['value']
    return field_dic


def save_FwFM_model_weights(model, model_weights_dir, categorical_feature_counts, feature_types, feature_dir, embed_dim):
    if not tf.gfile.Exists(model_weights_dir):
        tf.gfile.MkDir(model_weights_dir)
    num_cat_fields = 0
    num_features = []
    field_order = []
    for feature_name, feature_type in feature_types.items():
        if feature_name in {'label', 'tag'} or feature_type == 'NUMERIC':
            continue
        else:
            num_cat_fields += 1
            num_features.append(categorical_feature_counts[feature_name])
            field_order.append(feature_name)
    print('Number of categorical fields are: {}'.format(num_cat_fields))
    print('Number of features are: {}'.format(sum(num_features)))

    print('Saving model weights to {}'.format(model_weights_dir))

    with tf.gfile.Open(join(model_weights_dir, 'model.txt'), 'w+') as modelfile:
        #Write the header to the file
        modelfile.write('==== Header ====\n')
        modelfile.write('model_version_id: 100\n')
        modelfile.write('n_field: {}\n'.format(num_cat_fields))
        modelfile.write('has_bias: 0\n')
        modelfile.write('embedding_dim: {}\n'.format(embed_dim))
        modelfile.write('total_number_of_features: {}\n'.format(sum(num_features)))

        # Check number of interaction pairs (used for model_pruned version)
        wp = model.get_variable_value('w_p')
        num_pair = 0
        for i in range(len(wp)):
            if wp[i][0] != 0.0:
                num_pair += 1
        modelfile.write('total_number_of_interaction_pairs: {}\n'.format(num_pair))

        # Write the default weight of each field
        for feature_column, feature_type in feature_types.items():
            if feature_column in {'label', 'tag'} or feature_type == 'NUMERIC':
                continue
            else:
                default_vector = model.get_variable_value('v_%s' % feature_column)[0]
                modelfile.write('default_weights: {}\u0001{}\u0001'.format(feature_column, categorical_feature_counts[feature_column]))
                N = len(default_vector)
                for i in range(N - 1):
                    modelfile.write('{},'.format(default_vector[i]))
                modelfile.write('{}\n'.format(default_vector[-1]))

        # Write the model weights to the file
        modelfile.write('==== Model ====\n')
        global_interception = model.get_variable_value('b')
        modelfile.write('b1: {}\n'.format(global_interception[0]))

        linear_term = model.get_variable_value('w_l')
        field_scalar = model.get_variable_value('w_p')
        assert(len(field_scalar) == num_cat_fields * (num_cat_fields - 1) / 2)

        # First write the feature embedding
        for feature_column, feature_type in feature_types.items():
            if feature_column in {'label', 'tag'} or feature_type == 'NUMERIC':
                continue
            else:
                f_embedding = model.get_variable_value('v_%s' % feature_column)[1:]
                field_dic = generate_field_dic(feature_dir, feature_column)
                for ft_idx, wt in enumerate(f_embedding):
                    ft_idx += 1
                    f_value = field_dic[ft_idx]
                    #remove the prefix of field_name
                    m = len(feature_column)
                    if feature_column == 'subdomain' or feature_column == 'page_tld' or feature_column == 'app_name':
                        modelfile.write('{}\u0001{}: '.format(feature_column, f_value))
                    else:
                        modelfile.write('{}\u0001{}: '.format(feature_column, f_value[m+1:]))
                    N = len(wt)
                    for i in range(N-1):
                        modelfile.write('{},'.format(wt[i]))
                    modelfile.write('{}\n'.format(wt[-1]))

        #Then for linear embedding
        for i in range(len(field_order)):
            modelfile.write('w_l\u0001{}: '.format(field_order[i]))
            for j in range(embed_dim - 1):
                modelfile.write('{},'.format(linear_term[i * embed_dim + j][0]))
            modelfile.write('{}\n'.format(linear_term[i * embed_dim + embed_dim - 1][0]))

        #Finally for field interaction:
        num_pair = 0
        for i in range(len(field_order)):
            for j in range(i+1, len(field_order)):
                if field_scalar[num_pair][0] != 0.0:
                    modelfile.write('r\u0001{}\u0001{}: {}\n'.format(field_order[i], field_order[j], field_scalar[num_pair][0]))
                num_pair += 1


def save_FmFM_model_weights(model, model_weights_dir, categorical_feature_counts, feature_types, feature_dir,
                            embed_dim):
    if not tf.gfile.Exists(model_weights_dir):
        tf.gfile.MkDir(model_weights_dir)
    num_cat_fields = 0
    num_features = []
    field_order = []
    for feature_name, feature_type in feature_types.items():
        if feature_name in {'label', 'tag'} or feature_type == 'NUMERIC':
            continue
        else:
            num_cat_fields += 1
            num_features.append(categorical_feature_counts[feature_name])
            field_order.append(feature_name)
    field_order = sorted(field_order)
    print('Number of categorical fields are: {}'.format(num_cat_fields))
    print('Number of features are: {}'.format(sum(num_features)))

    print('Saving model weights to {}'.format(model_weights_dir))

    with tf.gfile.Open(join(model_weights_dir, 'model.txt'), 'w') as modelfile:
        # Write the header to the file
        modelfile.write('==== Header ====\n')
        modelfile.write('model_version_id: 100\n')
        modelfile.write('n_field: {}\n'.format(num_cat_fields))
        modelfile.write('has_bias: 0\n')
        modelfile.write('embedding_dim: {}\n'.format(embed_dim))
        modelfile.write('total_number_of_features: {}\n'.format(sum(num_features)))

        # Check number of interaction pairs (used for model_pruned version)
        # wp = model.get_variable_value('w_p')
        # num_pair = 0
        # for i in range(len(wp)):
        #     if wp[i][0] != 0.0:
        #         num_pair += 1
        # modelfile.write('total_number_of_interaction_pairs: {}\n'.format(num_pair))

        # Write the default weight of each field
        for feature_column, feature_type in feature_types.items():
            if feature_column in {'label', 'tag'} or feature_type == 'NUMERIC':
                continue
            else:
                default_vector = model.get_variable_value('v_%s' % feature_column)[0]
                modelfile.write('default_weights: {}\u0001{}\u0001'.format(feature_column,
                                                                           categorical_feature_counts[feature_column]))
                N = len(default_vector)
                for i in range(N - 1):
                    modelfile.write('{},'.format(default_vector[i]))
                modelfile.write('{}\n'.format(default_vector[-1]))

        # Write the model weights to the file
        modelfile.write('==== Model ====\n')
        global_interception = model.get_variable_value('b')
        modelfile.write('b1: {}\n'.format(global_interception[0]))

        # linear_term = model.get_variable_value('w_l')
        # field_matrices = model.get_variable_value('w_p')
        field_matrices = {}
        for idx_l, feat_l in enumerate(field_order):
            for idx_r, feat_r in enumerate(field_order):
                if idx_r <= idx_l:
                    continue
                w = model.get_variable_value('%s_%s' % (feat_l, feat_r))
                modelfile.write(
                    'r\u0001%s\u0001%s:\t%s\n' % (feat_l, feat_r, pickle.dumps(w)))
                field_matrices['%s X %s' % (feat_l, feat_r)] = w
        pickle.dump(field_matrices, open(join(model_weights_dir, 'field_matrices.pickle'), 'wb'))

        # Then for linear embedding
        # for i in range(len(field_order)):
        #     modelfile.write('w_l\u0001{}: '.format(field_order[i]))
        #     for j in range(embed_dim - 1):
        #         modelfile.write('{},'.format(linear_term[i * embed_dim + j][0]))
        #     modelfile.write('{}\n'.format(linear_term[i * embed_dim + embed_dim - 1][0]))

        #Then for linear embedding
        for i in range(len(field_order)):
            modelfile.write('w_\u0001{}:\t'.format(field_order[i]))
            modelfile.write('{}\n'.format(model.get_variable_value('w_%s' % field_order[i]).flatten() ) )

        # First write the feature embedding
        for feature_column, feature_type in feature_types.items():
            if feature_column in {'label', 'tag'} or feature_type == 'NUMERIC':
                continue
            else:
                f_embedding = model.get_variable_value('v_%s' % feature_column)[1:]
                field_dic = generate_field_dic(feature_dir, feature_column)
                for ft_idx, wt in enumerate(f_embedding):
                    ft_idx += 1
                    f_value = field_dic[ft_idx]
                    # remove the prefix of field_name
                    m = len(feature_column)
                    if feature_column == 'subdomain' or feature_column == 'page_tld' or feature_column == 'app_name':
                        modelfile.write('{}\u0001{}: '.format(feature_column, f_value))
                    else:
                        modelfile.write('{}\u0001{}: '.format(feature_column, f_value[m + 1:]))
                    N = len(wt)
                    for i in range(N - 1):
                        modelfile.write('{},'.format(wt[i]))
                    modelfile.write('{}\n'.format(wt[-1]))

def dense_to_sparse(dense_tensor, n_dim):
    zero_t = tf.zeros_like(dense_tensor)
    dense_final = tf.where(tf.greater_equal(dense_tensor, tf.ones_like(dense_tensor) * n_dim), zero_t, dense_tensor)
    indices = tf.to_int64(
        tf.transpose([tf.range(tf.shape(dense_tensor)[0]), tf.reshape(dense_final, [-1])]))
    values = tf.ones_like(dense_tensor, dtype=tf.float32)
    shape = [tf.shape(dense_tensor)[0], n_dim]
    return tf.SparseTensor(
        indices=indices,
        values=values,
        dense_shape=shape
    )


def LR(features, labels, mode, params):

    X = []
    W = []
    X_numeric = []
    W_numeric = []
    feat_cnt = params["categorical_feature_counts"]

    for f_name in features.keys():
        if f_name in {'label', 'tag'}: #Not used in model training
            continue
        elif params["feature_types"][f_name] == 'NUMERIC':
            X_numeric.append(features[f_name])
            w = tf.get_variable('w0_%s' % f_name, shape=[1, 2])
            W_numeric.append(w)
        else:
            sparse_t = dense_to_sparse(features[f_name], feat_cnt[f_name] + 1)
            X.append(sparse_t)
            w = tf.get_variable('w0_%s' % f_name, shape=[feat_cnt[f_name] + 1, 2])
            W.append(w)

    b = tf.get_variable('b', shape=[2])
    logits = b
    for i in range(len(X)):
        logits = logits + tf.sparse_tensor_dense_matmul(X[i], W[i])
    for i in range(len(X_numeric)):
        logits = logits + tf.matmul(tf.reshape(X_numeric[i], [-1, 1]), W_numeric[i])

    predicted_classes = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits)[:,1],
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Compute loss.
    labels = tf.cast(labels, tf.int64)
    one_hot_labels = tf.one_hot(labels, depth=2, dtype=tf.int64)

    class_weights = tf.constant(([1, 1]), dtype=tf.int64)
    sample_weights = tf.reduce_sum(tf.multiply(one_hot_labels, class_weights), 1)

    #loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits, weights=sample_weights)

    l2_loss = tf.nn.l2_loss(b) * params['l2_linear']
    for w in W:
        l2_loss += params['l2_linear'] * tf.nn.l2_loss(w)
    for w_numeric in W_numeric:
        l2_loss += params['l2_linear'] * tf.nn.l2_loss(w_numeric)

    # Compute evaluation metrics.
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predicted_classes,
                                   name='acc_op')
    auc = tf.metrics.auc(labels=labels,
                         predictions=tf.nn.softmax(logits)[:, 1],
                         name='auc_op')
    metric_orig_loss = tf.metrics.mean_tensor(loss,
                                              name='orig_loss_op')
    metric_l2_loss = tf.metrics.mean_tensor(l2_loss,
                                              name='l2_loss_op')
    metrics = {'accuracy': accuracy, 'auc': auc, 'orig_loss': metric_orig_loss, 'l2_loss': metric_l2_loss}

    loss += l2_loss

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdamOptimizer(params['learning_rate'])
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, eval_metric_ops=metrics)


def FM(features, labels, mode, params):
    # Create three fully connected layers.

    X = []
    W  = []
    V = []
    X_square = []
    XV = []
    XV_square = []
    feat_cnt = params["categorical_feature_counts"]

    for f_name in features.keys():
        if f_name in {'label', 'tag'} or params["feature_types"][f_name] == 'NUMERIC':
            continue
        else:
            # X.append(one_hot_dense)
            sparse_t = dense_to_sparse(features[f_name], feat_cnt[f_name] + 1)
            X.append(sparse_t)
            w = tf.get_variable('w_%s' % f_name, shape=[feat_cnt[f_name] + 1, 1])
            W.append(w)
            v = tf.get_variable('v_%s' % f_name, shape=[feat_cnt[f_name] + 1, params['latent_factor']])
            V.append(v)

    b = tf.get_variable('b', shape=[1])
    logits = b
    for i in range(len(X)):
        logits = logits + tf.sparse_tensor_dense_matmul(X[i], W[i])
        x_square = tf.SparseTensor(X[i].indices, tf.square(X[i].values), tf.to_int64(tf.shape(X[i])))
        X_square.append(x_square)
        xv = tf.sparse_tensor_dense_matmul(X[i], V[i])
        XV.append(xv)
        xv_square = tf.sparse_tensor_dense_matmul(x_square, tf.square(V[i]))
        XV_square.append(xv_square)
    p1 = XV[0]
    p2 = XV_square[0]
    for i in range(1, len(XV)):
        p1 = p1 + XV[i]
        p2 = p2 + XV_square[i]
    p_final = tf.reshape(0.5 * tf.reduce_sum(tf.square(p1) - p2, 1), [-1, 1])


    logits = logits + p_final

    y_prob = tf.sigmoid(logits)
    pred_class = tf.cast((y_prob >= 0.5), tf.bool)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': pred_class,
            'probabilities': y_prob,
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Compute loss.
    labels = tf.reshape(tf.cast(labels, tf.float32), [-1, 1])
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
    loss = tf.math.reduce_mean(loss)

    l2_loss = tf.nn.l2_loss(b) * params['l2_linear']
    for w in W:
        l2_loss += params['l2_linear'] * tf.nn.l2_loss(w)
    for v in V:
        l2_loss += params['l2_latent'] * tf.nn.l2_loss(v)


    # Compute evaluation metrics.
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=pred_class,
                                   name='acc_op1')
    auc = tf.metrics.auc(labels=labels,
                         predictions=y_prob,
                         name='auc_op1')

    metric_orig_loss = tf.metrics.mean(loss, name='orig_loss_op')
    metric_l2_loss = tf.metrics.mean(l2_loss, name='l2_loss_op')
    metrics = {'accuracy': accuracy, 'auc': auc, 'orig_loss': metric_orig_loss, 'l2_loss': metric_l2_loss}

    loss += l2_loss

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdamOptimizer(params['learning_rate'])
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, eval_metric_ops=metrics)


def FFM(features, labels, mode, params):
    X = []
    V = []
    W = []
    F = params['latent_factor']
    feat_cnt = params["categorical_feature_counts"]

    M = 0 #Number of categorical features
    for f_name in features.keys():
        if f_name in {'label', 'tag'} or params["feature_types"][f_name] == 'NUMERIC':
            continue
        else:
            M += 1
            # X.append(one_hot_dense)
            sparse_t = dense_to_sparse(features[f_name], feat_cnt[f_name] + 1)
            X.append(sparse_t)
            w = tf.get_variable('w_%s' % f_name, shape=[feat_cnt[f_name] + 1, 1]) #[Mi, 1]
            W.append(w) #linear term

    for f_name in features.keys():
        if params["feature_types"][f_name] == 'NUMERIC':
            continue
        else:
            v = tf.get_variable('v_%s' % f_name, shape=[feat_cnt[f_name] + 1, M * F]) #[Mi, M*F]
            V.append(v)

    # global interception
    b = tf.get_variable('b', shape=[1])

    #linear term
    xw = [tf.sparse_tensor_dense_matmul(X[i], W[i]) for i in range(M)]

    # field tensor list with M of [N, M * F]
    xv = [tf.sparse_tensor_dense_matmul(X[i], V[i]) for i in range(M)]

    # concact to matrix [N, M^2 * F]
    l = tf.concat([xv[i] for i in range(M)], 1)
    xw1 = tf.concat([xw[i] for i in range(M)], 1)

    #Create field index
    index_left = []
    index_right = []

    for i in range(M):
        for j in range(M):
            if i != j:
                index_left.append(i * M + j)
                index_right.append(j * M + i)

    # reshape l_ to [N, M^2, F]
    l_ = tf.reshape(l, [-1, M*M, F])
    l_left = tf.gather(l_, index_left, axis=1)
    l_right = tf.gather(l_, index_right, axis=1)
    # element-wise multiplication of [N, M(M-1), F]
    p_full = tf.multiply(l_left, l_right)
    # Reduce to [N, M(M-1)]
    p = tf.reduce_sum(p_full, 2)

    #Reduce to [N, 1]
    p = tf.reduce_sum(p, 1, keep_dims=True)

    logits = tf.reduce_sum(xw1, 1,  keep_dims=True) + b + p

    y_prob = tf.sigmoid(logits)
    pred_class = tf.cast((y_prob >= 0.5), tf.bool)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': pred_class,
            'probabilities': y_prob,
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Compute loss.
    labels = tf.reshape(tf.cast(labels, tf.float32), [-1, 1])
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
    loss = tf.math.reduce_mean(loss)

    l2_loss = 0
    for w in W:
        l2_loss += tf.nn.l2_loss(w) * params['l2_linear']
    for v in V:
        l2_loss += tf.nn.l2_loss(v) * params['l2_latent']

    # Compute evaluation metrics.
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=pred_class,
                                   name='acc_op1')
    auc = tf.metrics.auc(labels=labels,
                         predictions=y_prob,
                         name='auc_op1')

    metric_orig_loss = tf.metrics.mean(loss, name='orig_loss_op')
    metric_l2_loss = tf.metrics.mean(l2_loss, name='l2_loss_op')
    metrics = {'accuracy': accuracy, 'auc': auc, 'orig_loss': metric_orig_loss, 'l2_loss': metric_l2_loss}

    loss += l2_loss

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdamOptimizer(params['learning_rate'])
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, eval_metric_ops=metrics)


def FwFM(features, labels, mode, params):
    V = []
    xv_cat = []
    F = params['latent_factor']
    feat_cnt = params["categorical_feature_counts"]

    M = 0 #Number of categorical features
    for f_name in features.keys():
        if f_name in {'label', 'tag'} or params["feature_types"][f_name] == 'NUMERIC':
            continue
        else:
            M += 1
            sparse_t = dense_to_sparse(features[f_name], feat_cnt[f_name] + 1)
            v = tf.get_variable('v_%s' % f_name, dtype=tf.float32,
                                initializer=tf.random_normal(shape=[feat_cnt[f_name] + 1, F], mean=0.0, stddev=0.01))
            V.append(v)
            xv_cat.append(tf.sparse_tensor_dense_matmul(sparse_t, v))  # tf.nn.l2_normalize(v, axis=-1)))

    # linear term
    w_l = tf.get_variable('w_l', dtype=tf.float32,
                          initializer=tf.random_normal(shape=[M * F, 1], mean=0.0, stddev=1.0))
    # field scalar
    w_p = tf.get_variable('w_p', dtype=tf.float32,
                          initializer=tf.random_normal(shape=[int(M * (M-1)/2), 1], mean=0.0, stddev=1.0))
    # global interception
    b = tf.get_variable('b', shape=[1], dtype=tf.float32)

    # concact to matrix [N, M * F]
    l_cat = tf.concat([xv_cat[i] for i in range(M)], 1)

    l = l_cat
    index_left = []
    index_right = []

    for i in range(M):
        for j in range(i + 1, M):
                index_left.append(i)
                index_right.append(j)

    # reshape l_ to [N, M, F]
    l_ = tf.reshape(l, [-1, M, F])
    l_left = tf.gather(l_, index_left, axis=1)
    l_right = tf.gather(l_, index_right, axis=1)

    # element-wise multiplication of [N, M(M-1)/2, F]
    p_full = tf.multiply(l_left, l_right)
    # Reduce to [N, M(M-1)/2]
    p = tf.reduce_sum(p_full, 2)

    #Reduce to [N, 1]
    p = tf.matmul(p, w_p)

    logits = tf.matmul(l_cat, w_l) + b + p

    y_prob = tf.sigmoid(logits)
    pred_class = tf.cast((y_prob >= 0.5), tf.bool)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': pred_class,
            'probabilities': y_prob,
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Compute loss.
    labels = tf.reshape(tf.cast(labels, tf.float32), [-1, 1])

    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
    loss = tf.math.reduce_mean(loss)

    # L2_loss
    l2_loss = tf.nn.l2_loss(w_p) * params['l2_r'] \
              + tf.nn.l2_loss(w_l) * params['l2_linear'] \
              + sum([tf.nn.l2_loss(v) * params['l2_latent'] for v in V])

    # Compute evaluation metrics.
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=pred_class,
                                   name='acc_op1')
    auc = tf.metrics.auc(labels=labels,
                         predictions=y_prob,
                         name='auc_op1')

    metric_orig_loss = tf.metrics.mean(loss, name='orig_loss_op')
    metric_l2_loss = tf.metrics.mean(l2_loss, name='l2_loss_op')
    metrics = {'accuracy': accuracy, 'auc': auc, 'orig_loss': metric_orig_loss, 'l2_loss': metric_l2_loss}

    loss += l2_loss

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    # variables_to_restore = tf.contrib.get_variables_to_restore()
    optimizer = tf.train.AdamOptimizer(params['learning_rate'])
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, eval_metric_ops=metrics)


def FvFM(features, labels, mode, params):
    V = []
    xv_cat = []
    F = params['latent_factor']
    feat_cnt = params["categorical_feature_counts"]

    M = 0  # Number of categorical features
    for f_name in features.keys():
        if f_name in {'label', 'tag'} or params["feature_types"][f_name] == 'NUMERIC':
            continue
        else:
            M += 1
            sparse_t = dense_to_sparse(features[f_name], feat_cnt[f_name] + 1)
            v = tf.get_variable('v_%s' % f_name, dtype=tf.float32,
                                initializer=tf.random_normal(shape=[feat_cnt[f_name] + 1, F], mean=0.0, stddev=0.01))
            V.append(v)
            xv_cat.append(tf.sparse_tensor_dense_matmul(sparse_t, v))

    # linear term
    w_l = tf.get_variable('w_l', dtype=tf.float32,
                          initializer=tf.random_normal(shape=[M * F, 1], mean=0.0, stddev=1.0))
    # field scalar
    w_p = tf.get_variable('w_p', dtype=tf.float32,
                          initializer=tf.random_normal(shape=[int(M * (M - 1) / 2), 1], mean=0.0, stddev=1.0))
    # global interception
    b = tf.get_variable('b', shape=[1], dtype=tf.float32)

    # concact to matrix [N, M * F]
    l_cat = tf.concat([xv_cat[i] for i in range(M)], 1)

    l = l_cat
    index_left = []
    index_right = []

    for i in range(M):
        for j in range(i + 1, M):
            index_left.append(i)
            index_right.append(j)

    # reshape l_ to [N, M, F]
    l_ = tf.reshape(l, [-1, M, F])
    l_left = tf.gather(l_, index_left, axis=1)
    l_right = tf.gather(l_, index_right, axis=1)

    # element-wise multiplication of [N, M(M-1)/2, F]
    p_full = tf.multiply(l_left, l_right)
    p_full = tf.multiply(p_full, w_p)

    # Reduce to [N, M(M-1)/2]
    p = tf.reduce_sum(p_full, 2)

    # Reduce to [N, 1]
    p = tf.reduce_sum(p, 1, keepdims=True)

    logits = tf.matmul(l_cat, w_l) + b + p

    y_prob = tf.sigmoid(logits)
    pred_class = tf.cast((y_prob >= 0.5), tf.bool)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': pred_class,
            'probabilities': y_prob,
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Compute loss.
    labels = tf.reshape(tf.cast(labels, tf.float32), [-1, 1])
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
    loss = tf.math.reduce_mean(loss)

    # L2_loss
    l2_loss = tf.nn.l2_loss(w_l) * params['l2_linear'] \
                + tf.nn.l2_loss(w_p) * params['l2_r'] \
                + sum([tf.nn.l2_loss(v) * params['l2_latent'] for v in V])

    # Compute evaluation metrics.
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=pred_class,
                                   name='acc_op1')
    auc = tf.metrics.auc(labels=labels,
                         predictions=y_prob,
                         name='auc_op1')

    metric_orig_loss = tf.metrics.mean(loss, name='orig_loss_op')
    metric_l2_loss = tf.metrics.mean(l2_loss, name='l2_loss_op')
    metrics = {'accuracy': accuracy, 'auc': auc, 'orig_loss': metric_orig_loss, 'l2_loss': metric_l2_loss}

    loss += l2_loss

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdamOptimizer(params['learning_rate'])
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, eval_metric_ops=metrics)


def FmFM(features, labels, mode, params):
    # Create three fully connected layers.
    V = []
    W = []
    xv = {}

    feat_cnt = params["categorical_feature_counts"]
    feat_dims = params['feat_dims']
    print('Field Dim - {}'.format(feat_dims))
    feat_dim_sum = sum(feat_dims.values())
    cross_fields = params['cross_fields']

    for f_name in features.keys():
        if f_name in {'label', 'tag'} or params["feature_types"][f_name] == 'NUMERIC':
            continue
        sparse_t = dense_to_sparse(features[f_name], feat_cnt[f_name] + 1)
        v = tf.get_variable('v_%s' % f_name, dtype=tf.float32, initializer=tf.random_normal(
                                shape=[feat_cnt[f_name] + 1, feat_dims[f_name]], mean=0.0, stddev=0.01))
        V.append(v)
        xv[f_name] = tf.sparse_tensor_dense_matmul(sparse_t, v)

    # linear term
    w_l = tf.get_variable('w_l', dtype=tf.float32,
                          initializer=tf.random_normal(shape=[feat_dim_sum, 1], mean=0.0, stddev=1.0))

    b = tf.get_variable('b', shape=[1], dtype=tf.float32)

    # concact to matrix [N, M * F]
    l_cat = tf.concat(list(xv.values()), 1)

    l_left, l_right = [], []
    cross_fields_selected = []

    field_order = sorted(feat_dims.items(), key=lambda x: (-x[1], x[0]))
    for idx_l, (feat_l, dim_l) in enumerate(field_order):
        for idx_r, (feat_r, dim_r) in enumerate(field_order[idx_l+1:]):
            idx_r += (idx_l + 1)
            name = '%s_%s' % (feat_l, feat_r)
            name_alt = '%s_%s' % (feat_r, feat_l)
            if cross_fields and name not in cross_fields and name_alt not in cross_fields:
                continue
            cross_fields_selected.append(name)
            w = tf.get_variable(name, dtype=tf.float32, initializer=tf.random_normal(
                                    shape=[dim_l, dim_r], mean=0.0, stddev=0.01))
            W.append(w)
            l_left.append(tf.matmul(xv[feat_l], w))
            l_right.append(xv[feat_r])
    print(cross_fields_selected)
    l_left = tf.concat(l_left, 1)
    l_right = tf.concat(l_right, 1)

    p = tf.multiply(l_left, l_right)

    # Reduce to [N, 1]
    p = tf.reduce_sum(p, 1, keepdims=True)

    # Add the linear part and bias
    logits = tf.matmul(l_cat, w_l) + b + p

    y_prob = tf.sigmoid(logits)
    pred_class = tf.cast((y_prob >= 0.5), tf.bool)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': pred_class,
            'probabilities': y_prob,
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Compute loss.
    labels = tf.reshape(tf.cast(labels, tf.float32), [-1, 1])
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
    loss = tf.math.reduce_mean(loss)

    # L2_loss
    l2_loss = tf.nn.l2_loss(w_l) * params['l2_linear'] \
              + sum([tf.nn.l2_loss(v) * params['l2_latent'] for v in V]) \
              + sum([tf.nn.l2_loss(w) * params['l2_r'] for w in W])

    # Compute evaluation metrics.
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=pred_class,
                                   name='acc_op1')
    auc = tf.metrics.auc(labels=labels,
                         predictions=y_prob,
                         name='auc_op1')

    metric_orig_loss = tf.metrics.mean(loss, name='orig_loss_op')
    metric_l2_loss = tf.metrics.mean(l2_loss, name='l2_loss_op')
    metrics = {'accuracy': accuracy, 'auc': auc, 'orig_loss': metric_orig_loss, 'l2_loss': metric_l2_loss}

    loss += l2_loss

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    # variables_to_restore = tf.contrib.get_variables_to_restore()
    optimizer = tf.train.AdamOptimizer(params['learning_rate'])
    # optimizer = tf.train.AdagradOptimizer(0.2)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, eval_metric_ops=metrics)


def deepFwFM(features, labels, mode, params):
    # Create three fully connected layers.

    X = []
    V = []
    F = params['latent_factor']
    feat_cnt = params["categorical_feature_counts"]

    M_cat = 0 #Number of categorical features
    for f_name in features.keys():
        if f_name in {'label', 'tag'} or params["feature_types"][f_name] == 'NUMERIC':
            continue
        else:
            M_cat += 1
            sparse_t = dense_to_sparse(features[f_name], feat_cnt[f_name] + 1)
            X.append(sparse_t)
            v = tf.get_variable('v_%s' % f_name, initializer=tf.random_normal(shape=[feat_cnt[f_name] + 1, F], mean=0.0, stddev=0.01))
            # w = tf.get_variable('w_%s' % f_name, initializer=tf.random_normal(shape=[F, 1], mean=0.0, stddev=0.01))
            V.append(v)
    M = M_cat
    # linear term
    w_l = tf.get_variable('w_l', initializer=tf.random_normal(shape=[M_cat * F, 1], mean=0.0, stddev=1.0))

    # final activation layer
    w_f = tf.get_variable('w_f', initializer=tf.random_normal(shape=[M + int(M * (M-1)/2) + params['deep_dimension'], 1], mean=0.0, stddev=1.0))
    # global interception
    b = tf.get_variable('b', shape=[1])

    xv_cat = [tf.sparse_tensor_dense_matmul(X[i], V[i]) for i in range(M_cat)]

    # concact to matrix [N, M * F]
    l_cat = tf.concat([xv_cat[i] for i in range(M_cat)], 1)

    # linear part
    l_linear = tf.matmul(l_cat, w_l) #[N, 1]

    # second order interaction part
    l = l_cat
    index_left = []
    index_right = []

    for i in range(M):
        for j in range(i + 1, M):
                index_left.append(i)
                index_right.append(j)

    # reshape l_ to [N, M, F]
    l_ = tf.reshape(l, [-1, M, F])
    l_left = tf.gather(l_, index_left, axis=1)
    l_right = tf.gather(l_, index_right, axis=1)
    # element-wise multiplication of [N, M(M-1)/2, F]
    p_full = tf.multiply(l_left, l_right)
    # Reduce to [N, M(M-1)/2]
    p = tf.reduce_sum(p_full, 2)

    # Deep part
    deepmatrix1 = tf.get_variable('deepmatrix1', initializer=tf.random_normal(shape=[M_cat * F, params['deep_dimension']], mean=0.0, stddev=0.01))
    deepmatrix2 = tf.get_variable('deepmatrix2', initializer=tf.random_normal(shape=[params['deep_dimension'], params['deep_dimension']], mean=0.0, stddev=0.01))
    d1 = tf.matmul(l, deepmatrix1) # [N, deep_dim]
    d2 = tf.matmul(d1, deepmatrix2) # [N, deep_dim]

    # Combine
    l_final = tf.concat([l_linear, p, d2], 1) #[N, 1 + (M-1)*M/2 + deep_dim]

    logits = b + tf.matmul(l_final, w_f)



    y_prob = tf.sigmoid(logits)
    pred_class = tf.cast((y_prob >= 0.5), tf.bool)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': pred_class,
            'probabilities': y_prob,
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Compute loss.
    labels = tf.reshape(tf.cast(labels, tf.float32), [-1, 1])
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
    loss = tf.math.reduce_mean(loss)

    #l2_loss = 0
    l2_loss = (tf.nn.l2_loss(w_f) + tf.nn.l2_loss(w_l)) * params['l2_linear']
    for v in V:
        l2_loss += tf.nn.l2_loss(v) * params['l2_latent']
    l2_loss += (tf.nn.l2_loss(deepmatrix1) + tf.nn.l2_loss(deepmatrix2)) * params['l2_latent']

    # Compute evaluation metrics.
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=pred_class,
                                   name='acc_op1')
    auc = tf.metrics.auc(labels=labels,
                         predictions=y_prob,
                         name='auc_op1')

    metric_orig_loss = tf.metrics.mean(loss, name='orig_loss_op')
    metric_l2_loss = tf.metrics.mean(l2_loss, name='l2_loss_op')
    metrics = {'accuracy': accuracy, 'auc': auc, 'orig_loss': metric_orig_loss, 'l2_loss': metric_l2_loss}

    loss += l2_loss


    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdamOptimizer(params['learning_rate'])
    #optimizer = tf.train.AdagradOptimizer(0.2)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, eval_metric_ops=metrics)


def deepFmFM(features, labels, mode, params):
    # Create three fully connected layers.
    V = []
    W = []
    xv = {}
    # xw = []
    # F = params['latent_factor']

    feat_cnt = params["categorical_feature_counts"]

    feat_dim = {}
    if params['feat_dim']:
        feat_dim = params['feat_dim']
    else:
        for k, v in feat_cnt.items():
            feat_dim[k] = int(np.ceil(np.log2(v)/2) + 1)
    feat_dim_sum = sum(feat_dim.values())
    print('Field Dim - {}'.format(feat_dim))

    M_cat = 0  # Number of categorical features
    # M_num = 0
    for f_name in features.keys():
        if f_name in {'label', 'tag'} or params["feature_types"][f_name] == 'NUMERIC':
            continue
        M_cat += 1
        sparse_t = dense_to_sparse(features[f_name], feat_cnt[f_name] + 1)
        v = tf.get_variable('v_%s' % f_name, dtype=tf.float32, shape=[feat_cnt[f_name] + 1, feat_dim[f_name]])
        # w = tf.get_variable('w_%s' % f_name, dtype=tf.float32, initializer=tf.random_normal(
        #     shape=[feat_cnt[f_name] + 1, 1], mean=0.0, stddev=0.01))
        V.append(v)
        # W.append(w)
        xv[f_name] = tf.sparse_tensor_dense_matmul(sparse_t, v)
        # xw.append(tf.sparse_tensor_dense_matmul(sparse_t, w))

    print(features.keys())
    # M = M_num + M_cat
    # linear term
    # w_l = tf.get_variable('w_l', dtype=tf.float32,
    #                       initializer=tf.random_normal(shape=[feat_dim_sum, 1], mean=0.0, stddev=1.0))
    # field scalar
    # w_p = tf.get_variable('w_p', dtype=tf.float32,
    #                       initializer=tf.random_normal(shape=[int(M * (M - 1) / 2), F], mean=0.0, stddev=1.0))
    # global interception
    b = tf.get_variable('b', shape=[1], dtype=tf.float32)

    # concact to matrix [N, M * F]
    l_cat = tf.concat(list(xv.values()), 1)
    # w_cat = tf.concat(xw, 1)
    # w_cat = tf.concat(xw, 1)
    # l = l_cat

    l_left, l_right = [], []
    # interactions = []

    cross_dim_sum = 0
    field_order = sorted(feat_dim.items(), key=lambda x: x[1], reverse=True)
    for idx_l, (feat_l, dim_l) in enumerate(field_order):
        for idx_r, (feat_r, dim_r) in enumerate(field_order[idx_l+1:]):
            idx_r += (idx_l + 1)
            w = tf.get_variable('%s_%s' % (feat_l, feat_r), dtype=tf.float32, shape=[dim_l, dim_r])
            W.append(w)
            l_left.append(tf.matmul(xv[feat_l], w))
            l_right.append(xv[feat_r])
            cross_dim_sum += dim_r
    #         int_vec = tf.matmul(xv[feat_l], w)
    #         interactions.append(tf.reduce_sum(tf.multiply(int_vec, xv[feat_r]), 1, keepdims=True))

    # interactions = tf.concat(interactions, 1)
    l_left = tf.concat(l_left, 1)
    l_right = tf.concat(l_right, 1)

    p = tf.multiply(l_left, l_right)



    # Deep part
    deepmatrix1 = tf.get_variable('deepmatrix1', dtype=tf.float32, shape=[feat_dim_sum,feat_dim_sum])
    deep_b1 = tf.get_variable('deep_b1', dtype=tf.float32, shape=[feat_dim_sum])
    deepmatrix2 = tf.get_variable('deepmatrix2', dtype=tf.float32, shape=[feat_dim_sum, feat_dim_sum])
    deep_b2 = tf.get_variable('deep_b2', dtype=tf.float32, shape=[feat_dim_sum])
    d1 = tf.matmul(l_cat, deepmatrix1) + deep_b1 # [N, deep_dim]
    d1 = tf.nn.relu(d1)

    d2 = tf.matmul(d1, deepmatrix2) + deep_b2# [N, deep_dim]
    d2 = tf.nn.relu(d2)

    # final activation layer
    w_f = tf.get_variable('w_f', shape=[feat_dim_sum + cross_dim_sum, 1])

    # Reduce to [N, 1]
    # p = tf.reduce_sum(p, 1, keepdims=True)
    l_f = tf.concat([p, d2], 1)
    logits = tf.matmul(l_f, w_f) + b
    y_prob = tf.sigmoid(logits)
    pred_class = tf.cast((y_prob >= 0.5), tf.bool)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': pred_class,
            'probabilities': y_prob,
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Compute loss.
    labels = tf.reshape(tf.cast(labels, tf.float32), [-1, 1])
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
    loss = tf.math.reduce_mean(loss)

    # L2_loss
    l2_loss = (tf.nn.l2_loss(w_f) + tf.nn.l2_loss(deepmatrix1) + tf.nn.l2_loss(deepmatrix2)) * params['l2_linear'] \
              + sum([tf.nn.l2_loss(v) * params['l2_latent'] for v in V])
              # + sum([tf.nn.l2_loss(w) * params['l2_latent'] for w in W])

    # Compute evaluation metrics.
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=pred_class,
                                   name='acc_op1')
    auc = tf.metrics.auc(labels=labels,
                         predictions=y_prob,
                         name='auc_op1')

    metric_orig_loss = tf.metrics.mean(loss, name='orig_loss_op')
    metric_l2_loss = tf.metrics.mean(l2_loss, name='l2_loss_op')
    metrics = {'accuracy': accuracy, 'auc': auc, 'orig_loss': metric_orig_loss, 'l2_loss': metric_l2_loss}

    loss += l2_loss

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    # variables_to_restore = tf.contrib.get_variables_to_restore()
    optimizer = tf.train.AdamOptimizer(params['learning_rate'])
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, eval_metric_ops=metrics)


def DCN(features, labels, mode, params):
    V = []
    xv = {}

    feat_cnt = params["categorical_feature_counts"]
    feat_dims = params['feat_dims']
    feat_dim_sum = sum(feat_dims.values())
    print('Field Dim - {}'.format(feat_dims))

    M_cat = 0  # Number of categorical features
    for f_name in features.keys():
        if f_name in {'label', 'tag'} or params["feature_types"][f_name] == 'NUMERIC':
            continue
        M_cat += 1
        sparse_t = dense_to_sparse(features[f_name], feat_cnt[f_name] + 1)
        v = tf.get_variable('v_%s' % f_name, dtype=tf.float32, shape=[feat_cnt[f_name] + 1, feat_dims[f_name]])
        V.append(v)
        xv[f_name] = tf.sparse_tensor_dense_matmul(sparse_t, v)

    print(features.keys())

    # global interception
    b = tf.get_variable('b', shape=[1], dtype=tf.float32)

    # concact to matrix [N, M * F]
    l_cat = tf.concat(list(xv.values()), 1)

    # mat_list, for L2 reg
    mat_list = []

    # DNN part
    h_layer = l_cat
    for i in range(4):
        deep_mat = tf.get_variable('deep_mat_%d' % i, dtype=tf.float32, shape=[feat_dim_sum, feat_dim_sum])
        deep_bias = tf.get_variable('deep_b_%d' % i, dtype=tf.float32, shape=[feat_dim_sum])
        mat_list.extend([deep_mat, deep_bias])
        h_layer = tf.matmul(h_layer, deep_mat) + deep_bias  # [N, deep_dim]
        h_layer = tf.nn.relu(h_layer)

    # cross network
    y_cross_0 = tf.reshape(l_cat, shape=[-1, feat_dim_sum, 1])
    y_cross_i = tf.reshape(l_cat, shape=[-1, 1, feat_dim_sum])
    y_cross = l_cat

    for i in range(4):
        x0T_x_x1 = tf.matmul(y_cross_0, y_cross_i)
        cross_layer_w = tf.get_variable('cross_layer_w_%d' % i, dtype=tf.float32, shape=[feat_dim_sum, 1])
        cross_layer_b = tf.get_variable('cross_layer_b_%d' % i, dtype=tf.float32, shape=[1, 1, feat_dim_sum])
        mat_list.extend([cross_layer_b, cross_layer_w])
        y_cross_i = tf.add(tf.reshape(tf.matmul(x0T_x_x1, cross_layer_w), shape=[-1, 1, feat_dim_sum]), cross_layer_b)
        y_cross = tf.add(y_cross, tf.reshape(y_cross_i, shape=[-1, feat_dim_sum]))

    # final activation layer
    w_f = tf.get_variable('w_f', shape=[2 * feat_dim_sum, 1])
    mat_list.append(w_f)

    # Reduce to [N, 1]
    l_f = tf.concat([y_cross, h_layer], 1)
    logits = tf.matmul(l_f, w_f) + b
    y_prob = tf.sigmoid(logits)
    pred_class = tf.cast((y_prob >= 0.5), tf.bool)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': pred_class,
            'probabilities': y_prob,
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Compute loss.
    labels = tf.reshape(tf.cast(labels, tf.float32), [-1, 1])
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
    loss = tf.math.reduce_mean(loss)

    # L2_loss
    l2_loss = sum([tf.nn.l2_loss(v) * params['l2_linear'] for v in mat_list])\
              + sum([tf.nn.l2_loss(v) * params['l2_latent'] for v in V])

    # Compute evaluation metrics.
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=pred_class,
                                   name='acc_op1')
    auc = tf.metrics.auc(labels=labels,
                         predictions=y_prob,
                         name='auc_op1')

    metric_orig_loss = tf.metrics.mean(loss, name='orig_loss_op')
    metric_l2_loss = tf.metrics.mean(l2_loss, name='l2_loss_op')
    metrics = {'accuracy': accuracy, 'auc': auc, 'orig_loss': metric_orig_loss, 'l2_loss': metric_l2_loss}

    loss += l2_loss

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    # variables_to_restore = tf.contrib.get_variables_to_restore()
    optimizer = tf.train.AdamOptimizer(params['learning_rate'])
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, eval_metric_ops=metrics)


def build_custom_linear_classifier(model_dir, feature_names, feature_types, categorical_feature_counts, l2_linear,
                        l2_latent, l2_r, learning_rate, latent_dimension, model=None, feat_dims=None, cross_fields=None):
    Call_functions = {
        'LR': LR,
        'FM': FM,
        'FFM': FFM,
        'FwFM': FwFM,
        'FvFM': FvFM,
        'FmFM': FmFM,
        'deepFwFM': deepFwFM,
        'deepFmFM': deepFmFM,
        'DCN': DCN
    }
    config = tf.estimator.RunConfig(keep_checkpoint_max=1, save_checkpoints_steps=10000)

    estimator = tf.estimator.Estimator(
        model_fn=Call_functions[model],
        model_dir=model_dir,
        params={
            'feature_names': feature_names,
            'feature_types': feature_types,
            'categorical_feature_counts': categorical_feature_counts,
            'n_classes': 2,
            'latent_factor': latent_dimension,
            'l2_linear': l2_linear,
            'l2_latent': l2_latent,
            'l2_r': l2_r,
            'learning_rate': learning_rate,
            'deep_dimension': 200,
            'feat_dims': feat_dims,
            'cross_fields': cross_fields,
            'emb_dim': 128},
        config=config)

    return estimator

