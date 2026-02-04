import numpy as np
import copy

class MyBaggingClf:
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
            self.estimator.fit(X_sample, y_sample)
            self.estimators.append(estimator)

    def predict_proba(self, X):
        X = np.asarray(X)

        all_probas = []
        for estimator in self.estimators:
            if hasattr(estimator, 'predict_proba'):
                probas = estimator.predict_proba(X)
                if probas.shape[1] == 2:
                    all_probas.append(probas[:, 1])
                else:
                    all_probas.append(probas[:, 0])
            else:
                preds = estimator.predict(X)
                all_probas.append(preds)

        mean_proba = np.mean(all_probas, axis=0)
        return mean_proba

    def predict(self, X):
        probas = self.predict_proba(X)
        predictions = (probas > 0.5).astype(int)
        return predictions

    def __str__(self):
        return f'MyBaggingClf: n_estimators={self.n_estimators}, max_samples={self.max_samples}'