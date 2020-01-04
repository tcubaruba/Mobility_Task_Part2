import time
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from utils.models_utils import get_scores_for_cross_val, prepare_multiclass_target, prepare_binary_target

# multiple svc model definition
models_dict = {
    'rbf': SVC(kernel="rbf", gamma="auto"),  # Radial-basis function kernel (aka squared-exponential kernel).
    'sigmoid': SVC(kernel="sigmoid", gamma="auto"),
    # couldn't be parameterized to get reasonable results
    'linear': SVC(kernel="linear", gamma="auto"),
    'poly': SVC(kernel="poly", degree=3, gamma="auto"),  # infeasible to compute with my notebook
}

mode_list = ["WALK", "BICYCLE", "CAR", "BUS", "TRAM", "METRO", "TRAIN"]


def ensemble_classifier(X: pd.DataFrame, y: np.ndarray, kernel_name="linear"):
    print(f"*** SVC-kernal: {kernel_name} ***")

    # split data
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=34, shuffle=True)

    # get new y where 'Walk' is labeled as 1
    for mode in mode_list:
        y_test_binary = prepare_binary_target(y_test, [mode])
        y_train_binary = prepare_binary_target(y_train, [mode])

        # 1. binary classification of WALK - NON-WALK
        print()
        print(f"*** {mode} ***")

        # retrieve new model object from models_dict
        svc_model = models_dict[kernel_name]

        print(f"shape x_train: {y_train_binary.shape}")
        print(f"shape x_test: {y_test_binary.shape}")

        # train model
        svc_model.fit(x_train, y_train_binary)

        # compute confusion matrix:
        y_pred = svc_model.predict(x_test)
        c_matrix = confusion_matrix(y_test, y_pred)
        print(c_matrix)

        # remove as target mode classified objects from TEST data only
        true_index_list = list()

        for index, row in x_test.iterrows():
            data_sample = np.array(row).reshape(1, -1)
            prediction = svc_model.predict(data_sample)
            if prediction == 1:  # if prediction label is 1 -> walk data sample
                true_index_list.append(index)

        # scores = get_scores_for_cross_val(svc_model, X, y_w)
        # for score in scores:
        #     print(score, ": ", scores[score])

        y_test = y_test.drop(true_index_list)
        x_test = x_test.drop(true_index_list)


def single_classifier(X: pd.DataFrame, y: np.ndarray, kernel="linear"):
    y = prepare_multiclass_target(y)
    # retrieve model object from models_dict
    svc_model = models_dict[kernel]

    # split data
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=34, shuffle=True)
    print(f"shape x_train: {x_train.shape}")
    print(f"shape x_test: {x_test.shape}")

    print()
    print("*" * 15)
    print(f"MODEL: svc-{kernel}")

    start = time.time()

    # train model
    svc_model.fit(x_train, y_train)
    print(f"trained in {time.time() - start} sec")

    # predict test_data
    y_pred = svc_model.predict(x_test)
    print(f"trained in {time.time() - start} sec")

    # confusion matrix
    labels = np.unique(y_pred)
    c_matrix = confusion_matrix(y_test, y_pred, labels=labels)
    print(c_matrix)
    scores = get_scores_for_cross_val(svc_model, X, y)
    for score in scores:
        print(score, ": ", scores[score])
