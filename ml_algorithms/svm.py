import numpy as np
import random

class MySVM:
    def __init__(self, n_iter=10, learning_rate=0.001, C=1, sgd_sample=None, random_state=42):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.C = C
        self.sgd_sample = sgd_sample
        self.random_state = random_state

    def fit(self, X, y, verbose=False):
        random.seed(self.random_state)

        X = np.asarray(X)
        y = np.asarray(y)

        y = np.where(y == 0, -1, 1)

        n = X.shape[0]
        d = X.shape[1]

        self.weights = np.ones(d)
        self.b = 1

        loss_start = 0
        for i in range(n):
            xi = X[i]
            yi = y[i]
            prediction = np.dot(self.weights, xi) + self.b
            loss_start += max(0, 1 - yi * prediction)
        loss_start = np.sum(self.weights ** 2) + (1 / n) * loss_start

        if verbose:
            print(f'start | loss: {loss_start}')

        for iteration in range(1, self.n_iter + 1):
            if self.sgd_sample is not None:
                if isinstance(self.sgd_sample, float):
                    sample_size = int(n * self.sgd_sample)
                else:
                    sample_size = self.sgd_sample

                sample_rows_idx = random.sample(range(n), sample_size)
                X_batch = X[sample_rows_idx]
                y_batch = y[sample_rows_idx]
            else:
                X_batch = X
                y_batch = y


            for i in range(len(X_batch)):
                xi = X_batch[i]
                yi = y_batch[i]

                condition = yi * (np.dot(self.weights, xi) + self.b)

                if condition >= 1:
                    grad_w = 2 * self.weights
                    grad_b = 0
                else:
                    grad_w = 2 * self.weights - self.C*(yi * xi)
                    grad_b = -self.C*yi

                self.weights = self.weights - self.learning_rate * grad_w
                self.b = self.b - self.learning_rate * grad_b

            loss = 0
            for i in range(n):
                xi = X[i]
                yi = y[i]
                prediction = np.dot(self.weights, xi) + self.b
                loss += max(0, 1 - yi * prediction)
            loss = np.sum(self.weights ** 2) + self.C * (1 / n) * loss

            if verbose and iteration % verbose == 0:
                print(f'{iteration} | loss: {loss}')

    def predict(self, X):
        X = np.asarray(X)

        predictions = np.dot(X, self.weights) + self.b
        predictions = np.sign(predictions)

        predictions = np.where(predictions == -1, 0, 1)

        return predictions

    def get_coef(self):
        return (self.weights, self.b)

    def __str__(self):
        return f'MySVM class: n_iter={self.n_iter}, learning_rate={self.learning_rate}'