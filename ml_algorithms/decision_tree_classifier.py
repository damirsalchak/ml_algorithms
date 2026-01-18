import numpy as np


class MyTreeClf:
    def __init__(self, max_depth=5, min_samples_split=2, max_leafs=20, criterion='entropy'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.criterion = criterion  # 'entropy' или 'gini'
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

                if self.criterion == 'entropy':
                    gain = self.information_gain(y, left_y, right_y)
                else:  # gini
                    gain = self.gini_gain(y, left_y, right_y)

                if gain > best_gain:
                    best_gain = gain
                    best_col_idx = col_idx
                    best_split = split_value

        return best_col_idx, best_split, best_gain

    def entropy(self, y):
        unique, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        entropy_value = 0
        for p in probabilities:
            if p > 0:
                entropy_value -= p * np.log2(p)
        return entropy_value

    def gini(self, y):
        unique, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        gini_value = 1
        for p in probabilities:
            gini_value -= p * p
        return gini_value

    def information_gain(self, y, left_y, right_y):
        parent_entropy = self.entropy(y)
        n = len(y)
        n_left = len(left_y)
        n_right = len(right_y)

        if n_left == 0 or n_right == 0:
            return 0

        left_entropy = self.entropy(left_y)
        right_entropy = self.entropy(right_y)

        child_entropy = (n_left / n) * left_entropy + (n_right / n) * right_entropy
        return parent_entropy - child_entropy

    def gini_gain(self, y, left_y, right_y):
        parent_gini = self.gini(y)
        n = len(y)
        n_left = len(left_y)
        n_right = len(right_y)

        if n_left == 0 or n_right == 0:
            return 0

        left_gini = self.gini(left_y)
        right_gini = self.gini(right_y)

        child_gini = (n_left / n) * left_gini + (n_right / n) * right_gini
        return parent_gini - child_gini

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

    def predict_proba(self, X):
        X = np.asarray(X)
        probabilities = []

        for i in range(len(X)):
            prob = self._find_leaf_value(X[i], self.tree)
            probabilities.append(prob)

        return np.array(probabilities)

    def _find_leaf_value(self, row, node):
        if node['leaf']:
            return node['value']

        if row[node['feature']] <= node['split']:
            return self._find_leaf_value(row, node['left'])
        else:
            return self._find_leaf_value(row, node['right'])

    def predict(self, X):
        probabilities = self.predict_proba(X)

        predictions = []
        for prob in probabilities:
            if prob > 0.5:
                predictions.append(1)
            else:
                predictions.append(0)

        return np.array(predictions)

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
        return f'MyTreeClf class: max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, max_leafs={self.max_leafs}, criterion={self.criterion}'