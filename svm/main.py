import sys

import pandas as pd
import numpy as np
import preprocess.preprocess
from sklearn.model_selection import train_test_split

import svm.svm_helper as svm


pd.set_option('display.max_columns', None)
np.set_printoptions(threshold=sys.maxsize)

input_data = pd.read_csv('../final/all_segments.csv')
features = pd.read_csv('../final/all_features.csv')

data = input_data.copy()
data['key_col'] = data['trip'].astype(str).str.cat(data['segment'].astype(str), sep='_')
data = data.drop(columns=['trip', 'segment'])

features['key_col'] = features['trip'].astype(str).str.cat(features['segment'].astype(str), sep='_')
features = features.drop(columns=['trip', 'segment'])
cols_to_join = preprocess.preprocess.find_high_corr(threshold=0.3)
cols_to_join.append('key_col')
cols_to_join = np.array(cols_to_join)
features_to_join = features[cols_to_join]

data = pd.merge(data, features_to_join, on='key_col')

data = data.set_index('key_col')
data = data.dropna(axis=0)
target = data['mode']
data = data.drop(columns=['mode', 'time_ms'])

print(data.head())


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
o_svm = svm.Svm()
o_svm.fit(X_train, y_train)