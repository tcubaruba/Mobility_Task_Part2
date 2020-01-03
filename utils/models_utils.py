import pandas as pd
import numpy as np

from preprocess.preprocess import find_high_corr
from preprocess.preprocess import target_to_numerical

from sklearn.model_selection import cross_validate


def load_features_data(correlation_threshold=0.1):
    features = pd.read_csv('../final/all_features.csv')

    features['key_col'] = features['trip'].astype(str).str.cat(features['segment'].astype(str), sep='_')
    features = features.drop(columns=['trip', 'segment'])
    cols_to_join = find_high_corr(threshold=correlation_threshold, drop_mode=False)
    cols_to_join.append('key_col')
    cols_to_join = np.array(cols_to_join)
    features_to_join = features[cols_to_join]

    data = features_to_join
    data = data[~data.isin([np.nan, np.inf, -np.inf]).any(1)]

    y = data[['mode', 'key_col']]
    X = data.drop(columns='mode')

    y = y.set_index('key_col')
    X = X.set_index('key_col')
    return X, y


def prepare_multiclass_target(y):
    y = target_to_numerical(y, 'mode')
    y = np.ravel(y)
    return y


def prepare_binary_target(full_y, true_val):
    y = np.where(full_y.isin(true_val), 1.0, 0.0)
    y = np.ravel(y)
    return y


def get_scores_for_cross_val(model, X, y):
    scoring = ['accuracy', 'precision_macro', 'precision_weighted', 'recall_macro', 'recall_weighted', 'f1_macro',
               'f1_weighted']
    scores = cross_validate(model, X, y, scoring=scoring,
                            cv=5, return_train_score=True, n_jobs=-1)
    return scores

