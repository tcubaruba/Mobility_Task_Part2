import pandas as pd
import numpy as np
import sys
from sklearn.neural_network import MLPClassifier
from utils.models_utils import load_features_data
from utils.models_utils import prepare_multiclass_target
from utils.models_utils import get_scores_for_cross_val
from preprocess.preprocess import normalize_X

pd.set_option('display.max_columns', None)
np.set_printoptions(threshold=sys.maxsize)

X, y = load_features_data(correlation_threshold=0.2)
y = prepare_multiclass_target(y)

# normalize X
X = normalize_X(X)

model = MLPClassifier(hidden_layer_sizes=(100, 100, 50), activation='relu', solver='adam', verbose=False, max_iter=1000,
                      alpha=0.0001, batch_size=100, warm_start=True)

scores = get_scores_for_cross_val(model, X, y)
for score in scores:
    print(score, ": ", scores[score])


