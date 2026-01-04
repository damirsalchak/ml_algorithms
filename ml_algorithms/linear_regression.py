import numpy as np
import random


class MyLineReg():
    def __init__(self, n_iter=100, learning_rate=0.5, metric=None, reg=None, l1_coef=0, l2_coef=0, sgd_sample=None,
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
        return f'MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}'

    def fit(self, X, y, verbose=False):
        random.seed(self.random_state)

        X_array = np.asarray(X)
        y_array = np.asarray(y)

        X_array = np.hstack((np.ones((X_array.shape[0], 1)), X_array))
        n, d = X_array.shape

        self.weights = np.ones((d, 1))

        pred_start = X_array @ self.weights
        loss_start = np.mean((pred_start - y_array.reshape(-1, 1)) ** 2)

        if self.metric:
            method = getattr(self, self.metric)
        else:
            method = None

        if verbose:
            print(f'start|loss: {loss_start}|{method(y_array, pred_start)}')

        for i in range(self.n_iter):
            if self.sgd_sample is not None:
                if isinstance(self.sgd_sample, float):
                    sample_size = max(1, int(n * self.sgd_sample))  # avoid 0
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

            pred_batch = X_batch @ self.weights
            error_batch = pred_batch - y_batch.reshape(-1, 1)

            if self.reg == 'l1':
                grad = (2 / n_batch) * X_batch.T @ error_batch + self.l1_coef * np.sign(self.weights)
            elif self.reg == 'l2':
                grad = (2 / n_batch) * X_batch.T @ error_batch + 2 * self.l2_coef * self.weights
            elif self.reg == 'elasticnet':
                grad = (2 / n_batch) * X_batch.T @ error_batch + self.l1_coef * np.sign(
                    self.weights) + 2 * self.l2_coef * self.weights
            else:
                grad = (2 / n_batch) * X_batch.T @ error_batch

            if callable(self.learning_rate):
                lr = self.learning_rate(i)
            else:
                lr = self.learning_rate
            self.weights = self.weights - lr * grad

            pred_full = X_array @ self.weights
            loss_full = np.mean((pred_full - y_array.reshape(-1, 1)) ** 2)

            if self.reg == 'l1':
                loss_full += self.l1_coef * np.sum(np.abs(self.weights))
            elif self.reg == 'l2':
                loss_full += self.l2_coef * np.sum(self.weights ** 2)
            elif self.reg == 'elasticnet':
                loss_full += self.l1_coef * np.sum(np.abs(self.weights)) + self.l2_coef * np.sum(self.weights ** 2)

            if self.metric:
                metric_value = method(y_array, pred_full.flatten())
                self.best_score = metric_value

            if verbose and (i % verbose == 0 or i == 0):
                if self.metric:
                    print(f'{i} | loss: {loss_full} | {self.metric}: {metric_value}')
                else:
                    print(f'{i} | loss: {loss_full}')

    def get_coef(self):
        return self.weights[1:].flatten()

    def predict(self, X):
        X_array = np.asarray(X)
        X_array = np.hstack((np.ones((X_array.shape[0], 1)), X_array))
        return X_array @ self.weights

    def get_best_score(self):
        return self.best_score

    def mse(self, y, y_pred):
        return np.mean((y - y_pred) ** 2)

    def mae(self, y, y_pred):
        return np.mean(np.abs(y - y_pred))

    def rmse(self, y, y_pred):
        return np.sqrt(self.mse(y, y_pred))

    def mape(self, y, y_pred):
        return 100 / len(y) * np.sum(np.abs((y - y_pred) / y))

    def r2(self, y, y_pred):
        return 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))