import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import (
    Pipeline,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import (
    SVC,
)

from data_preprocess import create_labels, load_data
from model import FeatureExtractor

TRAIN_UTTERANCES_COUNT = [30] * 9
TEST_UTTERANCES_COUNTS = [31, 35, 88, 44, 29, 24, 40, 50, 29] 

def main():
    x_train = load_data("ae.train")
    y_train = create_labels(TRAIN_UTTERANCES_COUNT)
    x_test = load_data("ae.test")
    y_test = create_labels(TEST_UTTERANCES_COUNTS)

    ProjectPipeline = Pipeline([
        ("features", FeatureExtractor(segments=5, std= True)),
        ("scale_features", StandardScaler()),
        ("classifier", SVC(kernel="rbf", C= 0.1, gamma="scale")) #RBF kernel can create flexible, nonlinear decision boundaries
    ])

    
    param_grid = {
        "features__segments": [1, 2],
        "features__std": [True],
        "classifier__C": [1],
        "classifier__gamma": ["scale"],

    }

    cross_validation = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    search = GridSearchCV(ProjectPipeline, param_grid, cv=cross_validation, n_jobs=1)
    search.fit(x_train, y_train)

    test_accuracy = search.score(x_test, y_test)
    misclassifcation_percentage =  (1 - test_accuracy) * 100

    print("Cross Validation")

    for params, mean, std in zip(search.cv_results_["params"], search.cv_results_["mean_test_score"], search.cv_results_["std_test_score"]):
        segments = params["features__segments"]
        use_std = params["features__std"]

        feature_count = segments * 12
        if use_std:
            feature_count += 12

        print(f"{segments} / {feature_count} /"
              f"{mean * 100:.2f}% / {std * 100 :.2f}")

    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Misclassifcation percentage: {misclassifcation_percentage:.2f}%")

    predictions = search.predict(x_test)
    speakers = list(range(1,10))

    matrix = confusion_matrix(y_test, predictions, labels=speakers)

    print("Individual Speakers")

    for i, speaker in enumerate(speakers):
        correct = matrix[i, i]
        total = matrix[i].sum()
        accuracy = correct / total * 100

        print(f"{speaker} / {correct} / {total} / {accuracy:.2f}%")

    fig, ax = plt.subplots(figsize=(8, 7))

    display = ConfusionMatrixDisplay (confusion_matrix=matrix, display_labels=speakers)

    display.plot(ax=ax, values_format="d",colorbar=False)

    ax.set_xlabel("Predicted speaker")
    ax.set_ylabel("Actual speaker")
    ax.set_title("Speaker classification on the test set")
    fig.tight_layout()

    fig.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    main()
