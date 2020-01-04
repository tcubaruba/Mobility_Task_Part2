import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from utils.models_utils import prepare_multiclass_target
from utils.models_utils import get_scores_for_cross_val


def single_classifier_results(training_data: np.array, labels: np.array) -> dict:
    x = training_data
    y = pd.DataFrame.copy(labels)
    y = prepare_multiclass_target(y)

    dtc = DecisionTreeClassifier(criterion='gini')

    # measure accuracy of training
    scores = get_scores_for_cross_val(dtc, x, y)
    return scores
