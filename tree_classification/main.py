from tree_classification import single_classifier as sc
from tree_classification import ensemble_classifier as ec
from utils.models_utils import load_features_data


def print_scores(sc: dict):
    for s in sc:
        print(s, ": ", sc[s])
    print("\n")


if __name__ == '__main__':
    print("[=== Decision Tree based prediction of Mobility Data ===]")
    # setup training data
    X, y = load_features_data(correlation_threshold=0.1)  # training data

    print("[****** Decision Tree Classification: Multi-Class ******]")
    single_classifier_scores = sc.single_classifier_results(X, y)
    print_scores(single_classifier_scores)

    # print("==========================================")
    print("[****** Ensemble Walk Classification: Multi-Class ******]")
    binary_classifying_results, non_walk_predictions = ec.ensemble_classifier_resuts(X, y)
    print("[****** Binary {Walk - Non_Walk} Classification Scores ******]")
    print_scores(binary_classifying_results)
    print("[****** Classification of Non_Walk Labels ******]")
    print_scores(non_walk_predictions)
