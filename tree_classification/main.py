from tree_classification import single_classifier as sc
from tree_classification import ensemble_classifier as ec
from utils.models_utils import load_features_data


def print_scores(sc: dict):
    for s in sc:
        print(s, ": ", sc[s])


if __name__ == '__main__':
    print("[=== Decision Tree based prediction of Mobility Data ===]")
    # setup training data
    X, y = load_features_data(correlation_threshold=0.1)  # training data

    print("[Decision Tree Classification: Multi-Class]")
    single_classifier_scores = sc.single_classifier_results(X, y)
    print("* Mean Training Accuracy: %0.2f" % single_classifier_scores['test_accuracy'].mean())

    # print("==========================================")
    print("[Ensemble Walk Classification: Multi-Class]")
    binary_classifying_results, non_walk_predictions = ec.ensemble_classifier_resuts(X, y)
    print("* Mean Training Accuracy of Walk-Data: %0.2f" % binary_classifying_results['test_accuracy'].mean())
    print("* Mean Training Accuracy of Rail / Road Data: %0.2f" % non_walk_predictions['test_accuracy'].mean())
