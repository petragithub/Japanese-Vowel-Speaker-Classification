from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import (
    Pipeline,  # used for the case ofsequantial application of transformers: see further: 1. at first extract the features, 2. Scale those extracted features, 3. classify those scale dfeatures 
)
from sklearn.preprocessing import StandardScaler  #-> standard for SVC + RBF rekrnel 
from sklearn.svm import (
    SVC,  #-> choice is based on the fact that it is a standartd kernel fgor the SVVM classification task. We would like to measure the similarity between the two data points in infinite dimension and then approach the classification by the majkoptrity vote.
)

from data_preprocess import create_labels, load_data
from model import FeatureExtractor

# GidSearchCV 
# Generate test sets such that all contain the same distribution of classes, or as close as possible.

# Be invariant to class label: relabelling y = ["Happy", "Sad"] to y = [1, 0] should not change the indices generated.

#Preserve order dependencies in the dataset ordering, when shuffle=False: all samples from class k in some test set were contiguous in y, or separated in y by samples from classes other than k.

#Generate test sets where the smallest and largest differ by at most one sample.
# StratifieldkFold -> tyope of cross-validation  


TRAIN_UTTERANCES_COUNT = [30] * 9
TEST_UTTERANCES_COUNTS = [31, 35, 88, 44, 29, 24, 40, 50, 29] #Number of UTTERANCES per speaker according to https://archive.ics.uci.edu/dataset/128/japanese+vowels


def main():
    x_train = load_data("ae.train")
    y_train = create_labels(TRAIN_UTTERANCES_COUNT)
    x_test = load_data("ae.test")
    y_test = create_labels(TEST_UTTERANCES_COUNTS)

    ProjectPipeline = Pipeline([
        ("features", FeatureExtractor(segments=5, std= True)),
        ("scale_features", StandardScaler()),
        ("classifier", SVC(kernel="rbf", C= 0.1, gamma="scale")) #RBF kernel can create flexible, nonlinear decision boundaries. 
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

    print(test_accuracy)
    print(f"{misclassifcation_percentage:.2f}%")

if __name__ == "__main__":
    main()
