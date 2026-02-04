import numpy as np
import copy

class MyBaggingReg:
    def __init__(self, estimator=None, n_estimators=10, max_samples=1.0, random_state=42):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.estimators = []

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        np.random.seed(self.random_state)

        n_samples = X.shape[0]
        sample_size = int(n_samples * self.max_samples)

        self.estimators = []

        for i in range(self.n_estimators):
            indices = np.random.choice(n_samples, size=sample_size, replace=True)
            X_sample = X[indices]
            y_sample = y[indices]

            estimator = copy.deepcopy(self.estimator)
            estimator.fit(X_sample, y_sample)
            self.estimators.append(estimator)

    def predict(self, X):
        X = np.asarray(X)

        all_predictions = []
        for estimator in self.estimators:
            predictions = estimator.predict(X)
            all_predictions.append(predictions)

        mean_predictions = np.mean(all_predictions, axis=0)
        return mean_predictions

    def __str__(self):
        return f'MyBaggingReg: n_estimators={self.n_estimators}, max_samples={self.max_samples}'