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
    y_train = prepare_binary_target(y_train_orig, mode)
    y_test = prepare_binary_target(y_test_orig, mode)

    # normalize X
    X_train = normalize_X(X_train_orig)
    X_test = normalize_X(X_test_orig)

    model = MLPClassifier(hidden_layer_sizes=(50,), activation='relu', solver='adam', verbose=False, max_iter=1000,
                          alpha=0.0001,
                          batch_size=100, warm_start=True)

    scores = get_scores_for_cross_val(model, X_train, y_train)
    for score in scores:
        print(score, ": ", scores[score])

    model = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    conf = confusion_matrix(y_test, y_pred)
    print(conf)

    # remove walk data from X test
    true_index = list()
    true_i = list()
    for i, row in X_test.iterrows():
        data_sample = np.array(row).reshape(1, -1)
        prediction = model.predict(data_sample)
        # idx += 1
        if prediction == 1:  # if prediction label is 1 -> walk data sample
            # true_index.append(idx)
            true_i.append(i)

    # remove 'walk' predicated data from data set and corresponding label array
    # X_test = X_test.drop(X_test.index[[true_index]])
    # y_test = y_test_orig.drop(y_test_orig.index[[true_index]])
    # y_test = prepare_multiclass_target(y_test)

    X_test_false = X_test_orig.drop(true_i)
    y_test_false = y_test_orig.drop(true_i)
    X_test_true = X_test_orig.loc[true_i]
    y_test_true = y_test_orig.loc[true_i]

    # remove walk data from X train
    true_index = list()
    for i, _ in X_train.iterrows():
        for m in mode:
            if y_train_orig.loc[i, 'mode'] == m:  # if label is 1 -> walk data sample
                true_index.append(i)
                break

    # remove 'walk' predicated data from data set and corresponding label array
    # X_train = X_train.drop(X_train.index[[true_index]])
    # y_train = y_train_orig.drop(y_train_orig.index[[true_index]])
    # y_train = prepare_multiclass_target(y_train)

    X_train_false = X_train_orig.drop(true_index)
    y_train_false = y_train_orig.drop(true_index)
    X_train_true = X_train_orig.loc[true_index]
    y_train_true = y_train_orig.loc[true_index]

    return X_train, X_test, y_train, y_test, X_train_true, X_train_false, y_train_true, y_train_false, X_test_true, \
           X_test_false, y_test_true, y_test_false


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
X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(X, y_orig, test_size=0.2, random_state=42,
                                                                        shuffle=True)
print(y_test_orig['mode'].value_counts())
print(y_train_orig['mode'].value_counts())


print('-'*15, 'SEPARATIONG WALK AND NON-WALK', '-'*15)
X_train, X_test, y_train, y_test, X_train_true, X_train_false, y_train_true, y_train_false, X_test_true, \
           X_test_false, y_test_true, y_test_false = make_binary_classification(X_train_orig, X_test_orig, y_train_orig,
                                                                                y_test_orig, ['WALK'])
print(y_test_false['mode'].value_counts())
print(y_train_orig['mode'].value_counts())


# separate rail and road
print('-'*15, 'SEPARATIONG RAIL AND ROAD', '-'*15)
rail = ['TRAIN', 'METRO', 'TRAM']
road = ['BICYCLE', 'CAR', 'BUS']
# X_train, X_test, y_train, y_test, X_train_rail, X_train_road, y_train_rail, y_train_road, X_test_rail, \
#            X_test_road, y_test_rail, y_test_road = make_binary_classification(X_train_false, X_test_false,
#                                                                                 y_train_false, y_test_false, rail)
_, _, _, _, X_train_rail, X_train_road, y_train_rail, y_train_road, X_test_rail, \
           X_test_road, y_test_rail, y_test_road = make_binary_classification(X_train_orig, X_test_false,
                                                                                y_train_orig, y_test_false, rail)

_, X_test_false, _, y_test_false = X_train_rail, X_test_rail, y_train_rail, y_test_rail
print(y_test_false['mode'].value_counts())
print(y_train_orig['mode'].value_counts())
for mode in rail:
    print('-' * 15, 'SEPARATIONG ', mode.upper() +' AND NON-', mode.upper(), '-' * 15)
    # X_train, X_test, y_train, y_test, X_train_true, X_train_false, y_train_true, y_train_false, X_test_true, \
    # X_test_false, y_test_true, y_test_false = make_binary_classification(X_train_false, X_test_false,
    #                                                                             y_train_false, y_test_false, [mode])
    _, _, _, _, _, _, _, _, _, \
    X_test_false, _, y_test_false = make_binary_classification(X_train_orig, X_test_false,
                                                                       y_train_orig, y_test_false, rail)
    print(y_test_false['mode'].value_counts())
    print(y_train_orig['mode'].value_counts())

_, _,_, y_test_false = X_train_road, X_test_road, y_train_road, y_test_road
print(y_test_false['mode'].value_counts())
print(y_train_orig['mode'].value_counts())
for mode in road:
    print('-' * 15, 'SEPARATIONG ', mode.upper() +' AND NON-', mode.upper(), '-' * 15)

    # X_train, X_test, y_train, y_test, X_train_true, X_train_false, y_train_true, y_train_false, X_test_true, \
    # X_test_false, y_test_true, y_test_false = make_binary_classification(X_train_false, X_test_false,
    #                                                                             y_train_false, y_test_false, [mode])
    _, _, _, _, _, _, _, _, _, \
    X_test_false, _, y_test_false = make_binary_classification(X_train_orig, X_test_false,
                                                                                y_train_orig, y_test_false, [mode])
    print(y_test_false['mode'].value_counts())
    print(y_train_orig['mode'].value_counts())
