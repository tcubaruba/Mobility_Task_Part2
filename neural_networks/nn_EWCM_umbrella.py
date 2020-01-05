import pandas as pd
import numpy as np
import sys
from sklearn.neural_network import MLPClassifier
from utils.models_utils import load_features_data
from utils.models_utils import prepare_binary_target
from utils.models_utils import get_scores_for_cross_val
from preprocess.preprocess import normalize_X
from sklearn.model_selection import train_test_split
from utils.models_utils import get_final_metrics
import time

pd.set_option('display.max_columns', None)
np.set_printoptions(threshold=sys.maxsize)


def make_binary_classification(X_train_orig, X_test_orig, y_train_orig, y_test_orig, mode):
    y_train = prepare_binary_target(y_train_orig, mode)
    y_test = prepare_binary_target(y_test_orig, mode)

    # normalize X
    X_train = normalize_X(X_train_orig)
    X_test = normalize_X(X_test_orig)

    model = MLPClassifier(hidden_layer_sizes=(80, 60), activation='relu', solver='adam', verbose=False, max_iter=1000,
                          alpha=0.0001, batch_size=100, warm_start=True)

    scores = get_scores_for_cross_val(model, X_train, y_train)
    for score in scores:
        print(score, ": ", scores[score], " Average:", np.mean(scores[score]))

    model = model.fit(X_train, y_train)

    # get indices of trips classified as true in test set
    true_index = list()
    for i, row in X_test.iterrows():
        data_sample = np.array(row).reshape(1, -1)
        prediction = model.predict(data_sample)
        if prediction == 1:
            true_index.append(i)

    # separate "true" and "false" trips
    X_test_false = X_test_orig.drop(true_index)
    y_test_false = y_test_orig.drop(true_index)
    X_test_true = X_test_orig.loc[true_index]
    y_test_true = y_test_orig.loc[true_index]

    # same for train set
    true_index = list()
    for i, _ in X_train.iterrows():
        for m in mode:
            if y_train_orig.loc[i, 'mode'] == m:  # if label is 1 -> walk data sample
                true_index.append(i)
                break

    X_train_false = X_train_orig.drop(true_index)
    y_train_false = y_train_orig.drop(true_index)
    X_train_true = X_train_orig.loc[true_index]
    y_train_true = y_train_orig.loc[true_index]

    return X_train_true, X_train_false, y_train_true, y_train_false, X_test_true, \
           X_test_false, y_test_true, y_test_false


X, y_orig = load_features_data(correlation_threshold=0.05)
X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(X, y_orig, test_size=0.2, random_state=42,
                                                                        shuffle=True)
start = time.time()

print('-'*15, 'SEPARATIONG WALK AND NON-WALK', '-'*15)
X_train_true, X_train_false, y_train_true, y_train_false, X_test_true, \
           X_test_false, y_test_true, y_test_false = make_binary_classification(X_train_orig, X_test_orig, y_train_orig,
                                                                                y_test_orig, ['WALK'])
# result test set, add all trips which were classifies as WALK
res = X_test_true
res['mode'] = 'WALK'

# separate rail and road
print('-'*15, 'SEPARATIONG RAIL AND ROAD', '-'*15)
rail = ['TRAIN', 'METRO', 'TRAM']
road = ['BICYCLE', 'CAR', 'BUS']
X_train_rail, X_train_road, y_train_rail, y_train_road, X_test_rail, \
           X_test_road, y_test_rail, y_test_road = make_binary_classification(X_train_orig, X_test_false,
                                                                                y_train_orig, y_test_false, rail)

_, X_test_false, _, y_test_false = X_train_rail, X_test_rail, y_train_rail, y_test_rail
for mode in rail:
    print('-' * 15, 'SEPARATIONG ', mode.upper() +' AND NON-', mode.upper(), '-' * 15)
    _, _, _, _, X_test_true, X_test_false, _, y_test_false = make_binary_classification(X_train_orig, X_test_false,
                                                                       y_train_orig, y_test_false, [mode])
    # add classifies trips to result test set
    to_append = X_test_true
    to_append['mode'] = mode
    res = res.append(to_append)
# non-classifies trips
to_append_X = X_test_false
to_append_y = y_test_false

_, X_test_false,_, y_test_false = X_train_road, X_test_road, y_train_road, y_test_road
# add non-classified trips to ROAD
X_test_false = X_test_false.append(to_append_X)
y_test_false = y_test_false.append(to_append_y)
for mode in road:
    print('-' * 15, 'SEPARATIONG ', mode.upper() +' AND NON-', mode.upper(), '-' * 15)
    _, _, _, _, X_test_true, X_test_false, _, y_test_false = make_binary_classification(X_train_orig, X_test_false,
                                                                                y_train_orig, y_test_false, [mode])
    # add classifies trips to result test set
    to_append = X_test_true
    to_append['mode'] = mode
    res = res.append(to_append)

to_append = X_test_false
to_append['mode'] = 'WALK'  # set all trips which were not classified to walk
res = res.append(to_append)

y_pred = res['mode']
y_pred = y_pred.sort_index()
y_test_orig = y_test_orig.sort_index()

print(f"trained in {time.time() - start} sec")

scores = get_final_metrics(y_test_orig, y_pred)
print('-'*15, 'FINAL SCORES SINGLE CLASSIFIER MODEL NEURAL NETWORK', '-'*15)
for score in scores:
    print(score, ":\n", scores[score])

