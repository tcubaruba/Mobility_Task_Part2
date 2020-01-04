import pandas as pd
import numpy as np
import sys
from sklearn.neural_network import MLPClassifier
from utils.models_utils import load_features_data
from utils.models_utils import prepare_multiclass_target
from utils.models_utils import get_scores_for_cross_val
from preprocess.preprocess import normalize_X
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

pd.set_option('display.max_columns', None)
np.set_printoptions(threshold=sys.maxsize)

X, y_orig = load_features_data(correlation_threshold=0.05)
y = prepare_multiclass_target(y_orig)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, shuffle=True)

# normalize X
X_train = normalize_X(X_train)
X_test = normalize_X(X_test)

model = MLPClassifier(hidden_layer_sizes=(100, 100, 50), activation='relu', solver='adam', verbose=False, max_iter=1000,
                      alpha=0.0001, batch_size=100, warm_start=True)

scores = get_scores_for_cross_val(model, X_train, y_train)
for score in scores:
    print(score, ": ", scores[score])

model = model.fit(X_train, y_train)
y_pred = model.predict(X_test)
conf = confusion_matrix(y_test, y_pred)
print(conf)


