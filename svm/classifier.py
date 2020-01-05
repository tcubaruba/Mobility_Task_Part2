import time
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from utils.models_utils import get_scores_for_cross_val, prepare_multiclass_target, prepare_binary_target, \
    get_final_metrics

# multiple svc model definition
models_dict = {
    'rbf': SVC(kernel="rbf", gamma="auto", C=1.0, decision_function_shape='ovr'),
    'sigmoid': SVC(kernel="sigmoid", gamma="auto", C=1.0, decision_function_shape='ovr'),
    'linear': SVC(kernel="linear", gamma="auto", C=1.0, decision_function_shape='ovr'),
    'poly': SVC(kernel="poly", degree=3, gamma="auto", C=1.0, decision_function_shape='ovr'),
}

mode_list = ["WALK", "BICYCLE", "CAR", "BUS", "TRAM", "METRO", "TRAIN"]


def ensemble_classifier(X: pd.DataFrame, y: np.ndarray, kernel_name="linear"):
    print(f"*** ENSEMBLE CLASSIFIER ***")
    print(f"\tkernel: {kernel_name}")

    # split data
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

    print(f"\tshape x_train: {x_train.shape}")
    print(f"\tshape y_train: {y_train.shape}")

    start = time.time()

    # consecutive binary classification
    for mode in mode_list:
        print()
        print(f"*** {mode} ***")

        # prepare target vector (the matching mode is set to 1 all others to 0)
        y_test_binary = prepare_binary_target(y_test, [mode])
        y_train_binary = prepare_binary_target(y_train, [mode])

        # retrieve model object from models_dict
        svc_model = models_dict[kernel_name]

        print(f"\tshape x_test: {x_test.shape}")
        print(f"\tshape y_test: {y_test_binary.shape}")
        print()

        # train model
        svc_model.fit(x_train, y_train_binary)

        # compute confusion matrix:
        y_pred = svc_model.predict(x_test)
        c_matrix = confusion_matrix(y_test_binary, y_pred)
        print("confusion matrix: ")
        print(c_matrix)
        print()

        # print metrics
        print("scoring in all metrics: ")
        scoring = get_final_metrics(y_test_binary, y_pred)
        [print(f"\t{key}: {value}") for key, value in scoring.items()]
        print()

        # remove as target mode classified objects from TEST data only
        true_index_list = list()
        for index, row in x_test.iterrows():
            data_sample = np.array(row).reshape(1, -1)
            prediction = svc_model.predict(data_sample)
            if prediction == 1:  # if prediction label is 1 -> walk data sample
                true_index_list.append(index)
        y_test = y_test.drop(true_index_list)
        x_test = x_test.drop(true_index_list)
        print(f"*****" * 10)

    print(f"trained and predicted in {time.time() - start} sec")
    print()
    return


def single_classifier(X: pd.DataFrame, y: np.ndarray, kernel_name="linear"):
    print(f"*** SINGLE CLASSIFIER ***")
    print(f"\tkernel: {kernel_name}")

    # converts string target values to numbers
    y = prepare_multiclass_target(y)

    # retrieve model object from models_dict
    svc_model = models_dict[kernel_name]

    # split data
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    print(f"shape x_train: {x_train.shape}")
    print(f"shape y_train: {y_train.shape}")
    print(f"shape x_test: {x_test.shape}")
    print(f"shape y_test: {y_test.shape}")

    print()
    print("*" * 15)
    print(f"MODEL: svc-{kernel_name}")

    start = time.time()

    # train model
    svc_model.fit(x_train, y_train)

    # predict test_data
    y_pred = svc_model.predict(x_test)

    print(f"trained and predicted in {time.time() - start} sec")
    print()

    # confusion matrix
    print("confusion matrix: ")
    labels = np.unique(y_pred)
    c_matrix = confusion_matrix(y_test, y_pred, labels=labels)
    print(c_matrix)
    print()

    # print metrics
    print("scoring in all metrics: ")
    scoring = get_final_metrics(y_test, y_pred)
    print(scoring)
    print()

    # cross validation
    print("scores in cross validation:")
    scores = get_scores_for_cross_val(svc_model, X, y)
    for score in scores:
        print(score, ": ", scores[score])
    print()
    return
