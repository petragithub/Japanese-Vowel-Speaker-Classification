# Japanese Vowel Speaker Classification

## Project Overview

This project focuses on identifying speakers from short recordings of Japanese vowels using machine learning. The task is a nine-class classification problem in which each recording is assigned to one of nine male speakers.

The project uses statistical feature extraction combined with a Support Vector Machine (SVM) classifier. Since the recordings have different lengths, they are converted into fixed-length feature vectors before classification.

The main objective is to investigate whether dividing recordings into temporal segments improves classification accuracy compared with extracting statistical features from the entire recording.

## Dataset

The project uses the Japanese Vowels dataset from the UCI Machine Learning Repository. It contains recordings of nine male speakers pronouncing the Japanese vowel sequence /ae/.

Dataset link:
https://archive.ics.uci.edu/dataset/128/japanese+vowels

The dataset contains 640 recordings divided into:

- Training set (ae.train): 270 recordings, with 30 recordings per speaker.
- Test set (ae.test): 370 recordings, with different numbers of recordings per speaker.

Each recording is represented by a matrix of size T x 12, where T is the number of time steps and the 12 columns represent cepstral coefficients.

The recordings have variable lengths, generally between 10 and 30 time steps. In the data files, recordings are separated by rows containing twelve ones. These separator rows are removed during data loading.

The original training and testing split is preserved throughout the project.


## Requirements

The project uses Python 3.10 and the following main libraries:

- NumPy
- scikit-learn
- matplot
- ruff

The project dependencies are defined in pyproject.toml.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/petragithub/Japanese-Vowel-Speaker-Classification.git
cd Japanese-Vowel-Speaker-Classification
```

### 2. Install dependencies

Using uv:

```bash
uv sync
```

Alternatively, install the main Python dependencies using pip:

```bash
pip install numpy scikit-learn
```

## Running the Project

Run the main Python script from the repository directory.

Using uv:

```bash
uv run "my main.py"
```

Alternatively, using Python:

```bash
python "my main.py"
```

The script performs the following steps:

1. Loads the training and testing recordings.
2. Extracts statistical features from each recording.
3. Standardizes the extracted features.
4. Performs five-fold cross-validation.
5. Selects the best feature extraction configuration.
6. Retrains the selected pipeline on the complete training set.
7. Evaluates the classifier on the test set.
8. Outputs the final test accuracy and misclassification percentage.

## Methods

### Feature Extraction

Since the recordings have different lengths, statistical feature extraction is used to convert them into fixed-length vectors suitable for SVM classification.

Each recording is divided into a specified number of approximately equal temporal segments using NumPy's array_split function.

For every segment, the mean of each of the 12 cepstral coefficients is calculated. Additionally, the standard deviation of each coefficient is calculated over the complete recording.

Two configurations are evaluated:

| Segments | Features | Description |
|----------|----------|-------------|
| 1 | 24 | 12 overall means and 12 overall standard deviations |
| 2 | 36 | 24 segment means and 12 overall standard deviations |

Using two segments preserves some information about changes in the cepstral coefficients between the beginning and end of a recording.

### Feature Standardization

The extracted features are standardized using StandardScaler from scikit-learn.

Standardization prevents features with larger numerical values from having a disproportionate influence on the SVM classifier.

The scaler is included inside the classification pipeline. During cross-validation, it is fitted only on the training portion of each fold to prevent data leakage.

### SVM Classification

The classifier is implemented using scikit-learn's SVC with a Radial Basis Function (RBF) kernel.

The model uses the following parameters:

| Parameter | Value |
|-----------|-------|
| Classifier | Support Vector Machine |
| Kernel | RBF |
| Regularization parameter C | 1 |
| Gamma | scale |
| Multiclass strategy | One-vs-one |

The SVM parameters remain fixed throughout the experiments to focus on comparing the feature extraction configurations.

## Cross-Validation and Model Selection

Model selection is performed using GridSearchCV with stratified five-fold cross-validation.

The cross-validation procedure uses:

- Number of folds: 5
- Shuffling: Enabled
- Random seed: 0
- Evaluation metric: Classification accuracy
- Number of segments tested: 1 and 2
- Standard deviation features: Enabled
- SVM regularization parameter: C = 1
- Gamma: scale

The 270 training recordings are divided into five folds of 54 recordings each.

In each iteration, four folds are used for training and the remaining fold is used for validation.

Both feature extraction configurations are evaluated using the same cross-validation procedure.

The configuration with the highest mean validation accuracy is selected. The complete pipeline is then refitted using all 270 training recordings.

The separate test set is not used during model selection.

## Results

### Cross-Validation Results

| Segments | Features | Mean Accuracy | Standard Deviation |
|----------|----------|---------------|--------------------|
| 1 | 24 | 95.19% | 1.48% |
| 2 | 36 | 96.30% | 1.17% |

The two-segment configuration achieved a higher mean cross-validation accuracy of 96.30%, compared with 95.19% for the one-segment configuration.

This represents an improvement of 1.11 percentage points.

The standard deviation also decreased from 1.48% to 1.17%, indicating more consistent validation performance across the five folds.

Based on these results, the two-segment configuration was selected for final testing.

### Final Test Results

After model selection, the two-segment pipeline was refitted using all 270 training recordings and evaluated on the separate test set containing 370 recordings.

| Metric | Result |
|--------|--------|
| Selected segments | 2 |
| Number of features | 36 |
| Test accuracy | 97.84% |
| Misclassification percentage | 2.16% |
| Correct predictions | 362 |
| Incorrect predictions | 8 |

The results indicate that statistical feature extraction combined with an RBF SVM can effectively distinguish recordings from the nine speakers.

The cross-validation results are documented in the project report, while the main Python script prints the final test accuracy and misclassification percentage.

## Authors

This project was developed as part of the Neural Networks block project at the University of Groningen.

- Miruna Lungu - S5882206
- Andrejs Tupikins - S5607442
- Petra Nicoara - S5972876