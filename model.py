import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, std=True, segments=3):   #adjust number of segments to see if there is any difference in the results (shlud be mentionedi n the report as well)
        self.segments = segments
        self.std = std


    def fit(self, x, y):
        return self


    def transform(self, x):
        feature_vectors = []

        for num in x:
            parts = np.array_split(num, self.segments)
            features = []

            for part in parts:
                features.append(part.mean(axis=0))

            if self.std:
                features.append(num.std(axis=0))

            feature_vectors.append(np.concatenate(features))

        return np.array(feature_vectors)
