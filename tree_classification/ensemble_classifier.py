import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from utils.models_utils import prepare_binary_target, get_final_metrics, prepare_multiclass_target
from sklearn.model_selection import train_test_split
import time


def ensemble_classifier_resuts(training_data: np.array, labels: np.array) -> tuple:
    x = training_data
    y = pd.DataFrame.copy(labels)
    y = prepare_binary_target(y, ['WALK'])

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

    dtc = DecisionTreeClassifier(criterion='entropy')

    start = time.time()
    dtc.fit(x_train, y_train)
    print(f"trained in {time.time() - start} sec")

    y_pred = dtc.predict(x_test)

    binary_classifying_results = get_final_metrics(y_test, y_pred)

    walk_data_idx = list()
    idx = -1
    for _, row in x.iterrows():
        data_sample = np.array(row).reshape(1, -1)
        prediction = dtc.predict(data_sample)
        idx += 1
        if prediction == 1:  # if prediction label is 1 -> walk data sample
            walk_data_idx.append(idx)

    # remove 'walk' predicated data from data set and corresponding label array
    non_walk_data_to_learn = x.drop(x.index[[walk_data_idx]])
    y = pd.DataFrame.copy(labels)
    non_walk_labels = y.drop(y.index[[walk_data_idx]])

    # train on the non walk classified data
    # labels := {'TRAM', 'TRAIN', 'METRO', 'CAR', 'BUS' 'BICYCLE'}
    non_walk_labels = prepare_multiclass_target(non_walk_labels)

    x_train, x_test, y_train, y_test = train_test_split(non_walk_data_to_learn, non_walk_labels, test_size=0.2, random_state=1)

    start = time.time()
    dtc.fit(x_train, y_train)
    print(f"trained in {time.time() - start} sec")

    y_pred = dtc.predict(x_test)

    non_walk_predictions = get_final_metrics(y_test, y_pred)

    return binary_classifying_results, non_walk_predictions
