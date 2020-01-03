import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def find_high_corr(threshold=0.3, drop_mode=True):

    features = pd.read_csv('../final/all_features.csv')
    # delete cols with nan values
    features = features.dropna(axis=1)

    input_data = pd.read_csv('../final/all_segments.csv')

    trip_segment_mode = input_data[['trip', 'segment', 'mode']]
    trip_segment_mode = trip_segment_mode.drop_duplicates()
    trip_segment_mode.reset_index(inplace=True)

    features['mode'] = trip_segment_mode['mode']
    features['mode'] = features['mode'].astype('category')
    features['mode'] = features['mode'].cat.codes
    features = features.drop(columns='segment')

    corr = features.corr()
    mode_corr = corr['mode']

    high_corr = mode_corr[abs(mode_corr) > threshold]

    high_corr_cols = list(high_corr.index)
    print(high_corr_cols)

    if drop_mode:
        high_corr_cols.remove('mode')

    return high_corr_cols


def target_to_numerical(data, column):
    target = data[column]
    target = target.astype('category')
    target = target.cat.codes
    target = target.values
    return target


def normalize_X(orig_data):
    data = orig_data.copy()
    colnames = data.columns
    index = data.index
    data = pd.DataFrame(normalize(data), columns=colnames, index=index)
    return data

