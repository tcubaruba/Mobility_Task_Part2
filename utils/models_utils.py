import pandas as pd
import numpy as np

from preprocess.preprocess import find_high_corr
from preprocess.preprocess import target_to_numerical

from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix


def load_features_data(correlation_threshold=0.1):
    """
    Loads data with all features per trip
    :param correlation_threshold: select only features with minimum absolute correlation to mode
    :return: X, y
    """
    features = pd.read_csv('../final/all_features.csv')

    features['key_col'] = features['trip'].astype(str).str.cat(features['segment'].astype(str), sep='_')
    features = features.drop(columns=['trip', 'segment'])
    cols_to_join = find_high_corr(threshold=correlation_threshold, drop_mode=False)
    cols_to_join.append('key_col')
    cols_to_join = np.array(cols_to_join)
    features_to_join = features[cols_to_join]

    data = features_to_join
    data = data[~data.isin([np.nan, np.inf, -np.inf]).any(1)]

    y = data[['mode', 'key_col']]
    X = data.drop(columns='mode')

    y = y.set_index('key_col')
    X = X.set_index('key_col')
    X = pd.DataFrame(X)
    y = pd.DataFrame(y)
    return X, y


def prepare_multiclass_target(y):
    """
    Converts multiclass target vector from string to numerical
    :param y: target vector with string values
    :return: target vector with numerical values
    """
    y = target_to_numerical(y, 'mode')
    y = np.ravel(y)
    return y


def prepare_binary_target(full_y, true_classes):
    """
    Converts multi-class target vector to binary vector
    :param full_y: target array with multiple classes
    :param true_classes: array of classes which should be converted to True
    :return: binary target vector with Ones for true classes and Zeros for other classes
    """
    y = np.where(full_y.isin(true_classes), 1.0, 0.0)
    y = np.ravel(y)
    return y


def get_scores_for_cross_val(model, X, y):
    """
    Get cross validation scores for the model: 'accuracy', 'precision_macro', 'precision_weighted', 'recall_macro',
    'recall_weighted', 'f1_macro', 'f1_weighted'
    :param model: model to score
    :param X: Dataframe X
    :param y: target vector
    :return: dictionary with scores listed above
    """
    scoring = ['accuracy', 'precision_macro', 'precision_weighted', 'recall_macro', 'recall_weighted', 'f1_macro',
               'f1_weighted']
    scores = cross_validate(model, X, y, scoring=scoring,
                            cv=5, return_train_score=True, n_jobs=-1)
    return scores

def get_final_metrics(y_true, y_pred):
    """
    Get scores for test target vector and predictions vector: 'accuracy', 'precision_macro', 'precision_weighted',
    'recall_macro', 'recall_weighted', 'f1_macro', 'f1_weighted', 'confusion matrix'
    :param y_true: test target vector
    :param y_pred: predictions vector
    :return: dictionary with scores listed above
    """
    scoring = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision macro": precision_score(y_true, y_pred, average='macro'),
        "Precision weighted": precision_score(y_true, y_pred, average='weighted'),
        "Recall macro": recall_score(y_true, y_pred, average='macro'),
        "Recall weighted": recall_score(y_true, y_pred, average='weighted'),
        "F1 macro": f1_score(y_true, y_pred, average='macro'),
        "F1 weighted": f1_score(y_true, y_pred, average='weighted'),
        "Confusion matrix": confusion_matrix(y_true, y_pred)
    }
    return scoring

