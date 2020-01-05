import numpy as np
import pandas as pd
import time
from sklearn.tree import DecisionTreeClassifier
from utils.models_utils import prepare_multiclass_target, get_final_metrics
from sklearn.model_selection import train_test_split


def single_classifier_results(training_data: np.array, labels: np.array) -> dict:
    x = training_data
    y = pd.DataFrame.copy(labels)
    y = prepare_multiclass_target(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

    dtc = DecisionTreeClassifier(criterion='gini')
    start = time.time()
    dtc.fit(x_train, y_train)
    print(f"trained in {time.time() - start} sec")

    y_pred = dtc.predict(x_test)

    # measure accuracy of training
    scores = get_final_metrics(y_test, y_pred)
    return scores
