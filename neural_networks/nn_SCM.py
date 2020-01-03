import pandas as pd
import numpy as np
import sys
import preprocess.preprocess
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate
from utils.models_utils import load_features_data
from utils.models_utils import prepare_multiclass_target

pd.set_option('display.max_columns', None)
np.set_printoptions(threshold=sys.maxsize)

X, y = load_features_data()

y = prepare_multiclass_target(y)

# normalize X
X = preprocess.preprocess.normalize_X(X)

model = MLPClassifier(hidden_layer_sizes=(100,50), activation='relu', solver='adam', verbose=False, max_iter=1000, alpha=0.001)
                      #batch_size=64, alpha=0.001, early_stopping=True)

scoring = ['accuracy', 'precision_macro', 'precision_weighted', 'recall_macro', 'recall_weighted', 'f1_macro',
           'f1_weighted']
scores = cross_validate(model, X, y, scoring=scoring,
                        cv=5, return_train_score=True, n_jobs=-1)
# print(scores)
for score in scores:
    print(score, ": ", scores[score])


