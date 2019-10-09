import numpy as np
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.datasets import make_classification, make_regression, load_digits, load_boston
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error

RANDOM_STATE = 17


def entropy(y):
    e = 0
    for v in y:
        e += -v * np.log2(v)
    return e

def gini(y):
    sum = 0
    for i in y:
        sum += i ** 2
    return 1 - sum


def variance(y):
    return np.var(y)


def mad_median(y):
    pass


class Node():

    def __init__(self, feature_idx=0, threshold=0, labels=None, left=None, right=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.labels = labels
        self.left = left
        self.right = right

    def __str__(self):
        return 'feature index = {0},threshold ={1},[{2},{3}]'.format(self.feature_idx, self.threshold, len((self.left)), len((self.right)))

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

    def fit(self, X, y):

        if len(y) < self.min_samples_split:
            tree = DecisionTree()
            tree.root_node = Node(-1, -1, None, y, [0])
            tree.right_child = None
            tree.left_child = None
            return tree

        if X.shape[0] == 0:
            return None

        max_q = float("-inf")
        best_split_value = None
        best_feature = None

        for f in range(X.shape[1]):
            x_col = X[:, f]
            # if np.max(x_col) == np.min(x_col):
            #    continue
            splits = self._find_splits(x_col)

            # print('splits len {0}'.format(len(splits)))
            for spl in splits:

                # разбиваю данные по нацлучшему критерию
                q,_,_ = self._regression_var_criterion(X, f, y, spl)
                if q > max_q:
                    max_q = q
                    best_feature = f
                    best_split_value = spl

        if best_feature is None:
            return None

        X_left, X_right, y_l, y_r = self._get_data_split(X, y, best_feature, best_split_value)

        assert X_left is not None
        assert X_right is not None

        if self.max_depth == 0:
            tree = DecisionTree()
            tree.root_node = Node(best_feature, best_split_value, None, y_l, y_r)
            tree.left_child = None
            tree.right_child = None
            return tree

        new_depth = self.max_depth
        if new_depth != np.inf:
            new_depth -= 1
        print("call left branch with X_left.shape = {0}, len(y_l) = {1}".format(X_left.shape, len(y_l)))

        left_tree = DecisionTree(max_depth=new_depth, min_samples_split=self.min_samples_split,
                                 criterion=self.criterion)
        self.left_child = left_tree.fit(X_left, y_l)
        print("call right branch with X_right.shape = {0}, len(y_r) = {1}".format(X_right.shape, len(y_r)))
        right_tree = DecisionTree(max_depth=new_depth, min_samples_split=self.min_samples_split,
                                  criterion=self.criterion)
        self.right_child = right_tree.fit(X_right, y_r)

        self.root_node = Node(np.where(X.shape[1] == best_feature)[0], best_split_value, str(best_feature), y_l, y_r)

        return self

    def predict(self, X):

        predictions = []
        for x in X:

            cur_node = self.root_node
            leaf = cur_node
            while cur_node is not None:

                leaf = cur_node
                if x[cur_node.feature_idx] < cur_node.threshold:
                    cur_node = cur_node.left_child
                else:
                    cur_node = cur_node.right_child

                type = X.dtypes[leaf.feature_idx]
                if type == 'object':
                    predictions.append(self.most_frequent(np.append(leaf.left, leaf.right)))
                else:
                    predictions.append(np.mean(np.append(leaf.left, leaf.right)))
        return predictions

    def predict_proba(self, X):
        pass

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

    def most_frequent(List):
        return max(set(List), key=List.count)

    def _regression_var_criterion(self, X, feature, y, t):
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

    def _get_data_split(self, X, y, feature, threshold):

        x_col_values = X[:,feature]  #
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


X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)
print(X.shape)
dt = DecisionTree(criterion='gini', max_depth=3)
q = dt.fit(X_train, y_train)
print(q)
