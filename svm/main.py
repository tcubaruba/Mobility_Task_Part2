from utils.models_utils import load_features_data
from svm.classifier import single_classifier, ensemble_classifier

# data preparation
X, y = load_features_data(correlation_threshold=0.2)

single_classifier(X.copy(), y.copy(), 'linear')
ensemble_classifier(X.copy(), y.copy(), 'linear')
