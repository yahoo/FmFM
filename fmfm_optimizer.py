# Copyright Verizon Media
# This project is licensed under the MIT. See license in project root for terms.

import numpy as np
from os.path import join
from sklearn.decomposition import PCA
import json
import models
import features

def emb_dim_opt(model, feature_types, p_thres):
  # Embedding Dim Optimazer
  for feature_column, feature_type in feature_types.items():
    if feature_column in {'label', 'tag'} or feature_type == 'NUMERIC':
      continue
    else:
      mats = model.get_variable_value('v_%s' % feature_column)[1:]
      n_example, dim = mats.shape
      pca = PCA(n_components=min(dim, n_example), svd_solver='full')
      pca.fit(mats)
      ratio = pca.explained_variance_ratio_

      p = 0.
      for idx, r in enumerate(ratio):
        p += r
        if p > p_thres/100.:
          dim_opt = max(2, idx+1)
          print('%s\t%d -> %d' % (feature_column, dim, dim_opt))
          feat_dim[feature_column] = dim_opt
          break

  return feat_dim


def cross_terms_pruning(model, feat_dim, p_thres):
  # Mat Eigvals
  mat_eigvals = {}
  field_order = sorted(feat_dim.items(), key=lambda x: x[0])
  for idx_l, (feat_l, _) in enumerate(field_order):
    for idx_r, (feat_r, _) in enumerate(field_order[idx_l + 1:]):
      idx_r += (idx_l + 1)
      mat_name = '%s_%s' % (feat_l, feat_r)
      w = model.get_variable_value(mat_name)
      _, sigma, _ = np.linalg.svd(w)
      mat_eigvals[mat_name] = sigma.tolist()
  sorted_terms = sorted(mat_eigvals.items(), key=lambda x:np.var(x[1]), reverse=True)
  n_terms = len(sorted_terms) * p_thres // 100
  return [x[0] for x in sorted_terms[:n_terms]]


if __name__ == "__main__":
  # mode 1: optimize the embedding dimensions
  # mode 2: pruning the cross terms
  mode = 2
  model_dir = 'models/criteo_FmFM'
  data_dir = 'data/criteo'
  feature_meta = join(data_dir, 'features.json')
  feature_dict = join(data_dir, 'feature_index')

  feature_names, feature_defaults, categorical_feature_counts, feature_types, feat_dim = \
    features.build_feature_meta(feature_meta, feature_dict)

  model = models.build_custom_linear_classifier(
    model_dir, feature_names, feature_types, categorical_feature_counts,
    None, None, None, None, None, 'FmFM', None)

  if mode == 1:
    p_thres=95
    feature_meta_opt = join(data_dir, 'features_opt_p%d.json')
    feat_dim_opt = emb_dim_opt(model, feature_types, p_thres)
    feat_meta_list = []
    for feat_name, feat_type, _ in json.load(open(feature_meta)):
      feat_meta_list.append((feat_name, feat_type, feat_dim_opt[feat_name]))
    json.dump(feat_meta_list, open(feature_meta_opt % p_thres, 'w'), indent=2)

  elif mode == 2:
    p_thres=20
    cross_terms_file = join(data_dir, 'cross_fields_p%d.json')
    cross_terms = cross_terms_pruning(model, feat_dim, p_thres)
    json.dump(cross_terms, open(cross_terms_file % p_thres, 'w'), indent=2)

