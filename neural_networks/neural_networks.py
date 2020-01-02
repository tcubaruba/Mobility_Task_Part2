import pandas as pd
import numpy as np
from keras.models import Sequential
import preprocess.preprocess
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Flatten

import keras
import scipy.sparse
from sklearn.preprocessing import normalize
import sys

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

colnames = data.columns
index = data.index
data = pd.DataFrame(normalize(data), columns=colnames, index=index)

# convert data to 3d array of shape (samples, time steps, features)
data3d = np.array(list(data.groupby('key_col').apply(pd.DataFrame.as_matrix)))

target = target.loc[~target.index.duplicated(keep='first')]
# # for binary classification
# target = np.where(target == 'WALK', 1.0, 0.0)

# for multi-label classification
target = target.astype('category')
target = target.cat.codes
target = target.values


def permute_matrix(vector):
    indptr = range(vector.shape[0] + 1)
    ones = np.ones(vector.shape[0])
    permut = scipy.sparse.csr_matrix((ones, vector, indptr))
    return permut.toarray()


target = permute_matrix(target)

print(target.shape)
print(data3d.shape)

x_train = data3d[0:10000]
y_train = target[0:10000]
print(x_train.shape)
print(y_train.shape)
x_test = data3d[10000:]
y_test = target[10000:]
print(x_test.shape)
print(y_test.shape)


def evaluate_model(trainX, trainy, testX, testy):
    verbose, epochs, batch_size = 1, 3, 1000
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    print(n_timesteps)
    print(n_features)
    print(n_outputs)
    model = Sequential()
    # model.add(LSTM(100, input_shape=(n_timesteps, n_features)))
    model.add(Dense(200, input_shape=(n_timesteps, n_features)))
    model.add(Dropout(0.8))
    model.add(Dense(100, activation='sigmoid', kernel_initializer='RandomNormal'))
    model.add(Flatten())
    model.add(Dense(n_outputs, activation='softmax', kernel_initializer='RandomNormal'))
    sgd = keras.optimizers.SGD(learning_rate=0.01, momentum=0.8, nesterov=False)
    model.compile(loss = 'binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    # fit network
    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
    pred = model.predict(trainX)
    # evaluate model
    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=1)
    return accuracy, pred

acc, pred = evaluate_model(x_train, y_train, x_test, y_test)
print(acc)
print(pred)
print(acc)

