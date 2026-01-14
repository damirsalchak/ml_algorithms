import numpy as np
import random

class MyKNNReg:
    def __init__(self, k=3, metric='euclidean', weight='uniform'):
        self.k = k
        self.metric = metric
        self.weight  = weight

    def fit(self, X, y):
        self.X = np.asarray(X)
        self.y = np.asarray(y)
        self.train_size = self.X.shape

    def predict(self, X):
        X_test = np.asarray(X)
        predictions = []

        for test_row in X_test:
            if self.metric == 'euclidean':
                distances = np.sqrt(np.sum((self.X - test_row) ** 2, axis=1))

            elif self.metric == 'manhattan':
                distances = np.sum(np.abs(self.X - test_row), axis=1)

            elif self.metric == 'chebyshev':
                distances = np.max(np.abs(self.X - test_row), axis=1)

            elif self.metric == 'cosine':
                dot_product = np.sum(self.X * test_row, axis=1)
                norm_train = np.sqrt(np.sum(self.X ** 2, axis=1))
                norm_test = np.sqrt(np.sum(test_row ** 2))
                distances = 1 - (dot_product / (norm_train * norm_test))

            nearest_idx = np.argsort(distances)[:self.k]
            nearest_y = self.y[nearest_idx]
            nearest_d = distances[nearest_idx]

            if self.weight == 'uniform':
                weights = np.ones(self.k)
            elif self.weight == 'rank':
                ranks = np.arange(1, self.k + 1)
                weights = 1 / ranks
            elif self.weight == 'distance':
                weights = 1 / (nearest_d + 1e-15)


            pred = np.sum(weights * nearest_y) / np.sum(weights)

            predictions.append(pred)

        return np.array(predictions)

    def __str__(self):
        return f'MyKNNClf class: k={self.k}'
