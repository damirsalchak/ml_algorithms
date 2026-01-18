import numpy as np


class MyTreeReg:
    def __init__(self, max_depth=5, min_samples_split=2, max_leafs=20):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.leafs_cnt = 0
        self.tree = None

    def get_best_split(self, X, y):
        n_features = X.shape[1]
        best_gain = -1
        best_col_idx = None
        best_split = None

        for col_idx in range(n_features):
            column_values = X[:, col_idx]
            unique_values = np.sort(np.unique(column_values))
            for i in range(len(unique_values) - 1):
                split_value = (unique_values[i] + unique_values[i + 1]) / 2.0
                left_mask = column_values <= split_value
                right_mask = column_values > split_value

                left_y = y[left_mask]
                right_y = y[right_mask]

                gain = self.mse_gain(y, left_y, right_y)

                if gain > best_gain:
                    best_gain = gain
                    best_col_idx = col_idx
                    best_split = split_value

        return best_col_idx, best_split, best_gain

    def mse(self, y):
        if len(y) == 0:
            return 0
        mean = y.mean()
        return np.mean((y - mean) ** 2)

    def mse_gain(self, y, left_y, right_y):
        parent_mse = self.mse(y)
        n = len(y)
        n_left = len(left_y)
        n_right = len(right_y)

        if n_left == 0 or n_right == 0:
            return 0

        left_mse = self.mse(left_y)
        right_mse = self.mse(right_y)

        child_mse = (n_left / n) * left_mse + (n_right / n) * right_mse
        return parent_mse - child_mse

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        self.leafs_cnt = 0
        self.tree = self._build_tree(X, y, 0)

    def _build_tree(self, X, y, depth):
        node = {}

        if (depth >= self.max_depth or
                len(y) < self.min_samples_split or
                self.leafs_cnt >= self.max_leafs or
                len(np.unique(y)) == 1):
            node['leaf'] = True
            node['value'] = y.mean() if len(y) > 0 else 0
            self.leafs_cnt += 1
            return node

        feature_idx, split_value, gain = self.get_best_split(X, y)

        if gain <= 0:
            node['leaf'] = True
            node['value'] = y.mean() if len(y) > 0 else 0
            self.leafs_cnt += 1
            return node

        left_mask = X[:, feature_idx] <= split_value
        right_mask = X[:, feature_idx] > split_value

        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            node['leaf'] = True
            node['value'] = y.mean()
            self.leafs_cnt += 1
            return node

        node['leaf'] = False
        node['feature'] = feature_idx
        node['split'] = split_value
        node['left'] = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node['right'] = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return node

    def predict(self, X):
        X = np.asarray(X)
        predictions = []

        for i in range(len(X)):
            pred = self._find_leaf_value(X[i], self.tree)
            predictions.append(pred)

        return np.array(predictions)

    def _find_leaf_value(self, row, node):
        if node['leaf']:
            return node['value']

        if row[node['feature']] <= node['split']:
            return self._find_leaf_value(row, node['left'])
        else:
            return self._find_leaf_value(row, node['right'])

    def print_tree(self):
        if self.tree is None:
            print("Дерево не обучено")
            return

        self._print_node(self.tree)

    def _print_node(self, node, indent=0):
        spaces = " " * indent

        if node['leaf']:
            print(f"{spaces}leaf = {node['value']:.4f}")
        else:
            print(f"{spaces}feature_{node['feature']} > {node['split']}")
            self._print_node(node['left'], indent + 2)
            self._print_node(node['right'], indent + 2)

    def __str__(self):
        return f'MyTreeReg class: max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, max_leafs={self.max_leafs}'