import pickle
import os
import csv


# with open(os.path.join('plots', 'feature_importance', 'DD.pkl'), 'rb') as handle:
#     b = pickle.load(handle)
#     print(b)

data = []

with open(os.path.join('results', 'DD_best_features_single_cache_xgboost.pkl'), 'rb') as handle:
    b = pickle.load(handle)
    b['dataset'] = 'DD'
    data.append(b)

with open(os.path.join('results', 'ENZYMES_best_features_single_cache_xgboost.pkl'), 'rb') as handle:
    b = pickle.load(handle)
    b['dataset'] = 'ENZYMES'
    data.append(b)

with open(os.path.join('results', 'IMDB-BINARY_best_features_single_cache_xgboost.pkl'), 'rb') as handle:
    b = pickle.load(handle)
    b['dataset'] = 'IMDB-BINARY'
    data.append(b)

with open(os.path.join('results', 'IMDB-MULTI_best_features_single_cache_xgboost.pkl'), 'rb') as handle:
    b = pickle.load(handle)
    b['dataset'] = 'IMDB-MULTI'
    data.append(b)

with open(os.path.join('results', 'NCI1_best_features_single_cache_xgboost.pkl'), 'rb') as handle:
    b = pickle.load(handle)
    b['dataset'] = 'NCI1'
    data.append(b)

with open(os.path.join('results', 'PROTEINS_full_best_features_single_cache_xgboost.pkl'), 'rb') as handle:
    b = pickle.load(handle)
    b['dataset'] = 'PROTEINS_full'
    data.append(b)

with open(os.path.join('results', 'REDDIT-BINARY_best_features_single_cache_xgboost.pkl'), 'rb') as handle:
    b = pickle.load(handle)
    b['dataset'] = 'REDDIT-BINARY'
    data.append(b)

with open(os.path.join('results', 'REDDIT-MULTI-5K_best_features_single_cache_xgboost.pkl'), 'rb') as handle:
    b = pickle.load(handle)
    b['dataset'] = 'REDDIT-MULTI-5K'
    data.append(b)

with open(os.path.join('results', 'COLLAB_best_features_single_cache_xgboost.pkl'), 'rb') as handle:
    b = pickle.load(handle)
    b['dataset'] = 'COLLAB'
    data.append(b)

# with open(os.path.join('plots', 'feature_importance', f'COLLAB.pkl'), 'rb') as handle:
#     b = pickle.load(handle)
#     d = b.to_dict('records')[0]
#     d = sorted(d.items(), key=lambda x: x[1], reverse=True)
#     ldp_features = ['deg max', 'deg', 'deg min', 'deg mean', 'deg stddev']
#     imp = [x for x in d if x[0] not in ldp_features]
#     print(imp)

# with open(os.path.join('results', 'DD_rfe_results.pkl'), 'rb') as handle:
#     b = pickle.load(handle)
#     print(b)

# with open(os.path.join('results', 'IMDB-MULTI_rfe_results_fixed.pkl'), 'rb') as handle:
#     b = pickle.load(handle)
#     print(b['selected_features'])

keys = set(data[0].keys())

with open(os.path.join('results', 'importance_selection_results_xgboost.csv'), 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=keys, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(data)

# import numpy as np

# x = np.load('features_cache/DD_base.zst', allow_pickle=True)
# print(x.shape)
