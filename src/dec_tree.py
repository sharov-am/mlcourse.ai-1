import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.datasets import make_classification, make_regression, load_digits, load_boston
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error
#from scipy.stats import entropy

RANDOM_STATE = 17

def entropy(y):
    y = np.array(y)
    p = [len(y[y == k]) / len(y) for k in np.unique(y)]
    return -np.dot(p, np.log2(p))



def gini(y):
        p = [len(y[y == k]) / len(y) for k in np.unique(y)]
        return 1 - np.dot(p, p)


def variance(y):
    return np.var(y)


def mad_median(y):
    pass


node_counter = 0


class Node():

    def __init__(self, feature_idx=0, threshold=0, labels=None, left=None, right=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.labels = labels
        self.left = left
        self.right = right
        self.counter = -1

    def __str__(self):
        return 'feature index = {0},threshold ={1},[{2},{3}]'.format(self.feature_idx, self.threshold, len((self.left)),
                                                                     len((self.right)))


class DecisionTree(BaseEstimator):

    def __str__(self):

        if self.left_child is None:
            left = ""
        else:
            left = "\r\n" + str(self.left_child)

        if self.right_child is None:
            right = ""
        else:
            right = "\r\n" + str(self.right_child)

        if self.root_node is None:
            current = ""
        else:
            current = str(self.root_node)
        return current + left + right

    def __init__(self, max_depth=np.inf, min_samples_split=2,
                 criterion='gini', debug=False):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        if criterion == 'gini':
            self.crit = gini
        else:
            self.crit = entropy

        self.left_child = None
        self.right_child = None
        self.root_node = None
        self.debug = debug

    def fit(self, X, y):
        global node_counter

        assert not len(y) < self.min_samples_split, "len(y) < self.min_samples_split"

        max_q = float("-inf")

        best_split_value = 0  # default
        best_feature = 0  # default

        for f in range(X.shape[1]):
            x_col = X[:, f]
            splits = self._find_splits(x_col)
            if len(splits) == 0:
                continue

            for spl in splits:

                # разбиваю данные по нацлучшему критерию для данного признака
                q = self._regression_var_criterion(X, f, y, spl)
                if q > max_q:
                    max_q = q
                    best_feature = f
                    best_split_value = spl

        X_left, X_right, y_l, y_r = self._get_data_split(X, y, best_feature, best_split_value)

        assert X_left is not None
        assert X_right is not None

        if len(y_l) < self.min_samples_split or len(y_r) < self.min_samples_split:
            self.left_child = None
            self.right_child = None
            self.root_node = Node(best_feature, best_split_value, 'leaf', y_l, y_r)
            self.root_node.counter = node_counter
            node_counter += 1
            return self

        if self.max_depth == 0:
            self.root_node = Node(best_feature, best_split_value, 'leaf', y_l, y_r)
            self.root_node.counter = node_counter
            node_counter += 1
            self.left_child = None
            self.right_child = None
            return self

        new_depth = self.max_depth
        if new_depth != np.inf:
            new_depth -= 1

        if self.debug:
            print("call left branch with X_left.shape = {0}, len(y_l) = {1}".format(X_left.shape, len(y_l)))

        left_tree = DecisionTree(max_depth=new_depth, min_samples_split=self.min_samples_split,
                                 criterion=self.criterion, debug=self.debug)
        self.left_child = left_tree.fit(X_left, y_l)

        if self.debug:
            print("call right branch with X_right.shape = {0}, len(y_r) = {1}".format(X_right.shape, len(y_r)))

        right_tree = DecisionTree(max_depth=new_depth, min_samples_split=self.min_samples_split,
                                  criterion=self.criterion, debug=self.debug)
        self.right_child = right_tree.fit(X_right, y_r)

        self.root_node = Node(best_feature, best_split_value, str(best_feature), y_l, y_r)

        return self

    def predict(self, X):

        predictions = []
        for x in X:

            cur_tree = self
            leaf = cur_tree.root_node
            while cur_tree is not None:

                leaf = cur_tree.root_node

                assert leaf is not None, "leaf.root_node is not None"

                if x[cur_tree.root_node.feature_idx] < cur_tree.root_node.threshold:
                    cur_tree = cur_tree.left_child
                else:
                    cur_tree = cur_tree.right_child

            assert leaf is not None, "leaf is not None"
            #            assert leaf.labels == 'leaf', "leaf.labels =='leaf'"

            if isinstance(x, pd.DataFrame):
                type = X.dtypes[leaf.feature_idx]
                if type == 'object':
                    predictions.append(self.most_frequent(np.append(leaf.left, leaf.right)))
                else:
                    predictions.append(np.mean(np.append(leaf.left, leaf.right)))
            else:
                predictions.append(self._most_frequent(list(np.append(leaf.left, leaf.right))))
        return predictions

    def predict_leaf_number(self, x):
        predictions = []

        cur_tree = self
        leaf = cur_tree.root_node
        while cur_tree is not None:

            leaf = cur_tree.root_node

            assert leaf is not None, "leaf.root_node is not None"

            if x[cur_tree.root_node.feature_idx] < cur_tree.root_node.threshold:
                cur_tree = cur_tree.left_child
            else:
                cur_tree = cur_tree.right_child

        assert leaf is not None, "leaf is not None"
        assert leaf.labels == 'leaf', "leaf.labels =='leaf'"
        assert leaf.counter >= 0, "leaf.counter >= 0"
        return leaf.counter

    def predict_proba(self, X):
        K = self.count_leafs()
        result = []
        i = 0
        for x in X:
            num = self.predict_leaf_number(x)
            t = np.zeros(K)
            t[num] = 1
            result.append(t)
        return np.reshape(result, (X.shape[0], K))

    def _find_splits(self, X):
        """Find all possible split values."""
        split_values = set()

        # Get unique values in a sorted order
        x_unique = list(np.unique(X))
        for i in range(1, len(x_unique)):
            # Find a point between two values
            average = (x_unique[i - 1] + x_unique[i]) / 2.0
            split_values.add(average)

        return list(split_values)

    def _most_frequent(self, lst):
        return max(set(lst), key=lst.count)

    def _regression_var_criterion2(self, X, feature, y, t):
        x_col_values = X[:, feature]

        assert X.shape[0] == len(x_col_values), "X.shape[0] == len(x_col_values)"
        assert X.shape[0] == len(y), "X.shape[0] == len(y)"
        # print('feature = {0}, X.shape[1] = {1}'.format(feature,X.shape[1]))
        y_l = [y[i] for i, val in enumerate(x_col_values) if val < t]
        y_r = [y[i] for i, val in enumerate(x_col_values) if val >= t]
        q = np.var(y) - len(y_l) / len(y) * self.crit(y_l) - len(y_r) / len(y) * self.crit(y_r)

        assert len(y_l) + len(y_r) == len(y)
        assert len(y_l) != 0 and len(y_r) != 0, "len(y_l) == 0 or len(y_r) == 0"

        return q, y_l, y_r

    def _regression_var_criterion(self, X, feature_idx, y, threshold):
        y = np.array(y)
        mask = X[:, feature_idx] < threshold
        n_obj = X.shape[0]
        n_left = np.sum(mask)
        n_right = n_obj - n_left
        if n_left > 0 and n_right > 0:
            return self.crit(y) - (n_left / n_obj) * \
                   self.crit(y[mask]) - (n_right / n_obj) * \
                   self.crit(y[~mask])
        else:
            return 0

    def _get_data_split(self, X, y, feature, threshold):

        x_col_values = X[:, feature]  #
        assert X.shape[0] == len(x_col_values), "X.shape[0] == len(x_col_values)"
        assert X.shape[0] == len(y), "X.shape[0] == len(y)"
        # print('feature = {0}, X.shape[1] = {1}'.format(feature,X.shape[1]))
        y_l = [y[i] for i, val in enumerate(x_col_values) if val < threshold]
        y_r = [y[i] for i, val in enumerate(x_col_values) if val >= threshold]
        X_l = X[X[:, feature] < threshold]
        X_r = X[X[:, feature] >= threshold]

        assert len(y_l) + len(y_r) == len(y), "len(y_l) + len(y_r) == len(y)"
        assert X_l.shape[0] + X_r.shape[0] == X.shape[0], "X_l.shape[0] + X_r.shape[0] == X.shape[0]"
        return X_l, X_r, y_l, y_r

    def count_leafs(self):
        if self.root_node.labels == 'leaf':
            return 1
        else:
            sum = 0
            if self.left_child is not None:
                sum += self.left_child.count_leafs()
            if self.right_child is not None:
                sum += self.right_child.count_leafs()
            return sum


X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)
# dt = DecisionTree(criterion='gini')
# q = dt.fit(X_train, y_train)

# tree_params = {'max_depth': list(range(3, 11)), 'criterion': ['gini', 'entropy']}
# tree_grid = GridSearchCV(dt, tree_params, cv=5, scoring='accuracy')
# tree_grid.fit(X_train, y_train)
#
# from pprint import pprint
# pprint(vars(tree_grid))

fig = plt.figure()
plt.xlabel('Max depth')
plt.ylabel('Mean CV accuracy')

for crit in [ 'gini','entropy' ]:
    tree_params = {'max_depth': list(range(3, 11)), }
    tree_grid = GridSearchCV(DecisionTree(criterion=crit), tree_params, cv=5, scoring='accuracy')
    tree_grid.fit(X_train, y_train)
    plt.plot(tree_params['max_depth'], tree_grid.cv_results_['mean_test_score'],)


plt.show()

