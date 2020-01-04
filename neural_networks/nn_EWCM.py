import pandas as pd
import numpy as np
import sys
from sklearn.neural_network import MLPClassifier
from utils.models_utils import load_features_data
from utils.models_utils import prepare_binary_target
from utils.models_utils import get_scores_for_cross_val
from preprocess.preprocess import normalize_X
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from utils.models_utils import prepare_multiclass_target

pd.set_option('display.max_columns', None)
np.set_printoptions(threshold=sys.maxsize)


def make_binary_classification(X_train_orig, X_test_orig, y_train_orig, y_test_orig, mode):
    y_train = prepare_binary_target(y_train_orig, [mode])
    y_test = prepare_binary_target(y_test_orig, [mode])

    # normalize X
    X_train = normalize_X(X_train_orig)
    X_test = normalize_X(X_test_orig)

    model = MLPClassifier(hidden_layer_sizes=(50,), activation='relu', solver='adam', verbose=False, max_iter=1000,
                          alpha=0.0001,
                          batch_size=100, warm_start=True)

    scores, model = get_scores_for_cross_val(model, X_train, y_train)
    for score in scores:
        print(score, ": ", scores[score])

    model = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    conf = confusion_matrix(y_test, y_pred)
    print(conf)

    # remove walk data from X test
    walk_data_idx = list()
    idx = -1
    for i, row in X_test.iterrows():
        data_sample = np.array(row).reshape(1, -1)
        prediction = model.predict(data_sample)
        idx += 1
        if prediction == 1:  # if prediction label is 1 -> walk data sample
            walk_data_idx.append(idx)

    # remove 'walk' predicated data from data set and corresponding label array
    X_test = X_test.drop(X_test.index[[walk_data_idx]])
    y_test = y_test_orig.drop(y_test_orig.index[[walk_data_idx]])
    y_test = prepare_multiclass_target(y_test)

    # remove walk data from X train
    walk_data_idx = list()
    for i, _ in X_train.iterrows():
        if y_train_orig.loc[i, 'mode'] == 'WALK':  # if label is 1 -> walk data sample
            walk_data_idx.append(idx)

    # remove 'walk' predicated data from data set and corresponding label array
    X_train = X_train.drop(X_train.index[[walk_data_idx]])
    y_train = y_train_orig.drop(y_train_orig.index[[walk_data_idx]])

    y_train = prepare_multiclass_target(y_train)

    return X_train, X_test, y_train, y_test


def make_multiclass_classification(X_train, X_test, y_train, y_test):
    # predict other classes

    model = MLPClassifier(hidden_layer_sizes=(100, 100, 50), activation='relu', solver='adam', verbose=False,
                          max_iter=1000,
                          alpha=0.0001, batch_size=100, warm_start=True)

    scores = get_scores_for_cross_val(model, X_train, y_train)
    for score in scores:
        print(score, ": ", scores[score])

    model = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    conf = confusion_matrix(y_test, y_pred)
    print(conf)


X, y_orig = load_features_data(correlation_threshold=0.05)
X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(X, y_orig, test_size=0.33, random_state=42, shuffle=True)

X_train, X_test, y_train, y_test = make_binary_classification(X_train_orig, X_test_orig, y_train_orig, y_test_orig, 'WALK')


