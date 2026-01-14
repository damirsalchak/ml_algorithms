import numpy as np
import random

class MyKNNClf:
    def __init__(self, k=3, metric='euclidean', weight='uniform'):
        self.k = k
        self.train_size = None
        self.metric = metric
        self.weight = weight

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

            nearest_indices = np.argsort(distances)[:self.k]
            nearest_labels = self.y[nearest_indices]
            nearest_dists = distances[nearest_indices]

            if self.weight == 'uniform':
                weights = np.ones(self.k)
            elif self.weight == 'rank':
                ranks = np.arange(1, self.k + 1)
                weights = 1 / ranks
            elif self.weight == 'distance':
                weights = 1 / (nearest_dists + 1e-15)

            weight_class_0 = 0
            weight_class_1 = 0

            for i in range(self.k):
                if nearest_labels[i] == 1:
                    weight_class_1 += weights[i]
                else:
                    weight_class_0 += weights[i]

            if weight_class_1 >= weight_class_0:
                predictions.append(1)
            else:
                predictions.append(0)

        return np.array(predictions)

    def predict_proba(self, X):
        X_test = np.asarray(X)
        probabilities = []

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

            nearest_indices = np.argsort(distances)[:self.k]
            nearest_labels = self.y[nearest_indices]
            nearest_dists = distances[nearest_indices]

            if self.weight == 'uniform':
                weights = np.ones(self.k)
            elif self.weight == 'rank':
                weights = 1 / np.arange(1, self.k + 1)
            elif self.weight == 'distance':
                weights = 1 / (nearest_dists + 1e-15)

            sum_w_1 = 0
            for i in range(self.k):
                if nearest_labels[i] == 1:
                    sum_w_1 += weights[i]

            prob_1 = sum_w_1 / np.sum(weights)
            probabilities.append(prob_1)
        return np.array(probabilities)


    def __str__(self):
        return f'MyKNNClf class: k={self.k}'

