import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from utils.models_utils import prepare_binary_target
from utils.models_utils import get_scores_for_cross_val
from sklearn.model_selection import train_test_split


def ensemble_classifier_resuts(training_data: np.array, labels: np.array) -> tuple:
    x = training_data
    y = pd.DataFrame.copy(labels)
    y = prepare_binary_target(y, ['WALK'])

    dtc = DecisionTreeClassifier(criterion='gini')
    binary_classifying_results, _ = get_scores_for_cross_val(dtc, x, y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

    dtc = DecisionTreeClassifier(criterion='gini')
    dtc.fit(x_train, y_train)

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
    non_walk_labels = prepare_binary_target(non_walk_labels, ['TRAM', 'TRAIN', 'METRO', 'CAR', 'BUS' 'BICYCLE'])

    # x_train, x_test, y_train, y_test = train_test_split(non_walk_data_to_learn, non_walk_labels, test_size=0.3, random_state=1)

    non_walk_predictions, _ = get_scores_for_cross_val(dtc, non_walk_data_to_learn, non_walk_labels)

    return binary_classifying_results, non_walk_predictions
