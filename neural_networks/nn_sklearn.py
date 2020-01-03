import pandas as pd
import numpy as np
import sys
import preprocess.preprocess
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import normalize

pd.set_option('display.max_columns', None)
np.set_printoptions(threshold=sys.maxsize)

features = pd.read_csv('../final/all_features.csv')

features['key_col'] = features['trip'].astype(str).str.cat(features['segment'].astype(str), sep='_')
features = features.drop(columns=['trip', 'segment'])
cols_to_join = preprocess.preprocess.find_high_corr(threshold=0.1, drop_mode=False)
cols_to_join.append('key_col')
cols_to_join = np.array(cols_to_join)
features_to_join = features[cols_to_join]

data = features_to_join
data = data[~data.isin([np.nan, np.inf, -np.inf]).any(1)]

y = data[['mode', 'key_col']]
X = data.drop(columns='mode')

y = y.set_index('key_col')
X = X.set_index('key_col')

# # only walk/non-walk
# y = np.where(y == 'WALK', 1.0, 0.0)

# convert target to numerical values
y = preprocess.preprocess.target_to_numerical(y, 'mode')

# normalize X
X = preprocess.preprocess.normalize_X(X)

model = MLPClassifier(hidden_layer_sizes=(100,50), activation='relu', solver='adam', verbose=True)
fit = model.fit(X,y)
pred = model.predict(X)
score = model.score(X,y)

print(pred[:100])
print(y[:100])
print(score)
