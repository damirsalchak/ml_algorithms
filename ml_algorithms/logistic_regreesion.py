import numpy as np
import random


class MyLogReg():
    def __init__(self, n_iter=10, learning_rate=0.1, metric=None, reg=None, l1_coef=0, l2_coef=0, sgd_sample=None,
                 random_state=42):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.metric = metric
        self.weights = []
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.sgd_sample = sgd_sample
        self.random_state = random_state
        self.best_score = None

    def __str__(self):
        return f'MyLogReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}'

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y, verbose=False):
        random.seed(self.random_state)
        X_array = np.asarray(X)
        y_array = np.asarray(y).reshape(-1, 1)

        X_array = np.hstack((np.ones((X_array.shape[0], 1)), X_array))
        n, d = X_array.shape

        self.weights = np.ones((d, 1))

        z_start = X_array @ self.weights
        pred_start = self.sigmoid(z_start)

        eps = 1e-15
        loss_start = -np.mean(y_array * np.log(pred_start + eps) + (1 - y_array) * np.log(1 - pred_start + eps))

        if self.metric:
            method = getattr(self, self.metric)
        else:
            method = None

        if verbose:
            score_str = f"|{method(y_array, (pred_start > 0.5).astype(int))}" if method else ""
            print(f'start|loss: {loss_start}{score_str}')

        for i in range(1, self.n_iter + 1):
            if self.sgd_sample is not None:
                if isinstance(self.sgd_sample, float):
                    sample_size = max(1, int(n * self.sgd_sample))
                else:
                    sample_size = self.sgd_sample
                sample_rows_idx = random.sample(range(n), sample_size)
                X_batch = X_array[sample_rows_idx]
                y_batch = y_array[sample_rows_idx]
                n_batch = X_batch.shape[0]
            else:
                X_batch = X_array
                y_batch = y_array
                n_batch = n

            z_batch = X_batch @ self.weights
            pred_batch = self.sigmoid(z_batch)
            error_batch = pred_batch - y_batch

            grad = (1 / n_batch) * X_batch.T @ error_batch

            if self.reg == 'l1':
                grad += self.l1_coef * np.sign(self.weights)
            elif self.reg == 'l2':
                grad += 2 * self.l2_coef * self.weights
            elif self.reg == 'elasticnet':
                grad += self.l1_coef * np.sign(self.weights) + 2 * self.l2_coef * self.weights

            lr = self.learning_rate(i) if callable(self.learning_rate) else self.learning_rate
            self.weights -= lr * grad

            pred_full = self.sigmoid(X_array @ self.weights)
            loss_full = -np.mean(y_array * np.log(pred_full + eps) + (1 - y_array) * np.log(1 - pred_full + eps))

            if self.reg == 'l1':
                loss_full += self.l1_coef * np.sum(np.abs(self.weights))
            elif self.reg == 'l2':
                loss_full += self.l2_coef * np.sum(self.weights ** 2)
            elif self.reg == 'elasticnet':
                loss_full += self.l1_coef * np.sum(np.abs(self.weights)) + self.l2_coef * np.sum(self.weights ** 2)

            if self.metric:
                y_pred_binary = (pred_full > 0.5).astype(int)
                metric_value = method(y_array, y_pred_binary)
                self.best_score = metric_value

            if verbose and (i % verbose == 0 or i == 1):
                if self.metric:
                    print(f'{i} | loss: {loss_full} | {self.metric}: {metric_value}')
                else:
                    print(f'{i} | loss: {loss_full}')

    def get_coef(self):
        return self.weights[1:].flatten()

    def predict_proba(self, X):
        X_array = np.asarray(X)
        X_array = np.hstack((np.ones((X_array.shape[0], 1)), X_array))
        return self.sigmoid(X_array @ self.weights).flatten()

    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(int)

    def get_best_score(self):
        return self.best_score

    def accuracy(self, y, y_pred):
        return np.mean(y.flatten() == y_pred.flatten())

    def precision(self, y, y_pred):
        tp = np.sum((y.flatten() == 1) & (y_pred.flatten() == 1))
        fp = np.sum((y.flatten() == 0) & (y_pred.flatten() == 1))
        return tp / (tp + fp) if (tp + fp) > 0 else 0

    def recall(self, y, y_pred):
        tp = np.sum((y.flatten() == 1) & (y_pred.flatten() == 1))
        fn = np.sum((y.flatten() == 1) & (y_pred.flatten() == 0))
        return tp / (tp + fn) if (tp + fn) > 0 else 0

    def f1(self, y, y_pred):
        p = self.precision(y, y_pred)
        r = self.recall(y, y_pred)
        return 2 * p * r / (p + r) if (p + r) > 0 else 0