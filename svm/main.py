import sys
import time

import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC

import preprocess.preprocess
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import normalize

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
target = data[['mode']].to_numpy()
data = data.drop(columns=['mode', 'time_ms'])

colnames = data.columns
index = data.index
data = pd.DataFrame(normalize(data), columns=colnames, index=index)

# for binary classification
target = np.where(target == 'WALK', 1, 0)

print(f"shape X total: {data.shape}")
print(f"shape y total: {target.shape}")

x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.99, random_state=42)
print(f"shape X train: {x_train.shape}")
print(f"shape y train: {y_train.shape}")
print(f"shape X test: {x_test.shape}")
print(f"shape y test: {y_test.shape}")

start = time.time()

# train the model
svc = LinearSVC()
svc.fit(x_train, np.ravel(y_train))

duration_train = time.time() - start
print(f"trained in {duration_train} sec")
duration_classification = (time.time() - start) - duration_train

# predict
y_pred = svc.predict(x_test)
print(f"classified in {duration_classification} sec")
print(f"f1 score: {f1_score(y_test, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None)}")
