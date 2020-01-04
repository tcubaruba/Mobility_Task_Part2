import sys
import time

import pandas as pd
import numpy as np
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from utils.models_utils import load_features_data
from utils.models_utils import prepare_multiclass_target
from utils.models_utils import get_scores_for_cross_val
from preprocess.preprocess import normalize_X

# data preparation
X, y = load_features_data(correlation_threshold=0.2)
y = prepare_multiclass_target(y)

# split data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=34, shuffle=True)
# print(f"shape x_train: {x_train.shape}")
# print(f"shape x_test: {x_test.shape}")

# multiple svc model definition
models_dict = {
    'rbf': SVC(kernel="rbf", gamma="auto"),  # Radial-basis function kernel (aka squared-exponential kernel).
    'sigmoid': SVC(kernel="sigmoid", gamma="auto", decision_function_shape='ovo'),  # couldn't be parameterized to get reasonable results
    'linear': SVC(kernel="linear", gamma="auto"),
    # 'poly': SVC(kernel="poly", degree=3, gamma="auto"),  # infeasible to compute with my notebook
}

for svc_name in models_dict:
    print()
    print("*" * 15)
    print(f"MODEL: svc-{svc_name}")

    start = time.time()

    # train model
    svc_model = models_dict[svc_name]
    svc_model.fit(x_train, y_train)
    print(f"trained in {time.time() - start} sec")

    # predict test_data
    y_pred = svc_model.predict(x_test)
    print(f"trained in {time.time() - start} sec")

    # confusion matrix
    labels = np.unique(y_pred)
    c_matrix = confusion_matrix(y_test, y_pred, labels=labels)
    print(c_matrix)
    # scores = get_scores_for_cross_val(svc_model, X, y)
    # for score in scores:
    #     print(score, ": ", scores[score])
