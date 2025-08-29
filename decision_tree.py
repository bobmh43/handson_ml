import types
import itertools
import warnings

import numpy as np
import sklearn
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.linear_model import SGDRegressor, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, log_loss
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.exceptions import ConvergenceWarning
import matplotlib.pyplot as plt

class _Node:
    def __init__(self, *, depth):
        self.fidx = None # the index of X's features, starting from 0
        self.threshold = None
        self.left = None
        self.right = None
        self.depth = depth
        
        # leaf node attributes
        self.samples = None # number of training instances this node applies to
        self.value = None # bin count of the classes of the training instances this node applies to
        self.class_ = None # the predicted class of this node 

def _Gini(y):
    '''Gini impurity of a categorical array'''

    _, counts = np.unique(y, return_counts=True)
    return 1 - np.sum((counts / counts.sum()) ** 2)

def _Entropy(y):
    _, counts = np.unique(y, return_counts=True)
    # - sigma p log p
    proportions = counts / counts.sum()
    return -sum(proportions * np.log2(proportions))


# DEVELOPMENT PROCESS 
# 1. Layout the structure of MyTreeClassifier, __Node and the training algorithm
# 2. start with a simple version of fit
# 3. added the entropy impurity measure
# 4. added in all the other hyperparameters (a lot of work)
# 5. decided that build tree should be a method of MyTreeClassifier (it is still recursive on nodes)
# 6. added in the feature_importances_ attribute
class MyTreeClassifier(ClassifierMixin, BaseEstimator):
    '''
    Implements the CART algorithm for classification
    '''
    def __init__(self, 
                 criterion="gini",
                 splitter="best",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 max_features=None,
                 random_state=None,
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.0):
        '''
        `criterion`: {"gini", "entropy"}
        `splitter`: {"best", "random"}
        '''
        if criterion.lower() == "gini":
            self.impurity_measure = _Gini
        elif criterion.lower() == "entropy":
            self.impurity_measure = _Entropy
        else:
            raise ValueError("Unrecognized impurity measure: " + criterion)
        
        if splitter.lower() == "best":
            self.use_rand_thresh = False
        elif splitter.lower() == "random":
            self.use_rand_thresh = True  # is a extra randomized tree
        else:
            raise ValueError("Unrecognized splitter value: " + splitter)
        
        if max_depth is None:
            self.max_depth = np.inf
        else:
            self.max_depth = max_depth

        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

        self.max_leaf_nodes = max_leaf_nodes if max_leaf_nodes is not None else np.inf
        self.min_impurity_decrease = min_impurity_decrease

        self._fitted = False

    def _build_tree(self, node, X, y, indices):
        '''
        Takes a `node` and builds the decision tree with that as the root. returns None.\n
        `self` is used as a storage for all the regularization hyperparameters
        '''
        # setup
        lowest_impurity = np.inf
        best_left_indices = None
        best_right_indices = None
        split_found = False
        big_impurity_drop = False
        node.impurity = self.impurity_measure(y[indices])

        # a stop condition
        if self.n_leaf_nodes >= self.max_leaf_nodes:
            return
        
        # go condition
        if len(indices) >= self.min_samples_split and node.depth < self.max_depth:
            # for a random subset of the feature indices
            for features_seen, fidx in enumerate(np.random.permutation(X.shape[1])):
                if features_seen >= self.max_features and split_found and big_impurity_drop:
                    break
                ## an option for extremely randomized trees
                if self.use_rand_thresh:                          
                    possible_thresholds = [np.random.choice(X[indices, fidx])]
                else:                                                   
                    possible_thresholds = X[indices, fidx]

                for threshold in possible_thresholds:
                    # split
                    mask = X[indices, fidx] <= threshold
                    left_indices = indices[mask]
                    right_indices = indices[~mask]

                    ## we only consider this split if the resulting children can be leaves.
                    if len(left_indices) < self.min_samples_leaf or len(right_indices) < self.min_samples_leaf:
                        continue
                    else:
                        split_found = True

                    # find the impurities
                    left_impurity = self.impurity_measure(y[left_indices])
                    right_impurity = self.impurity_measure(y[right_indices])
                    impurity = len(left_indices) / len(indices) * left_impurity + len(right_indices) / len(indices) * right_impurity

                    # recording the current lowest
                    if impurity < lowest_impurity:
                        lowest_impurity = impurity
                        best_left_indices = left_indices
                        best_right_indices = right_indices
                        node.fidx = fidx
                        node.threshold = threshold
                        big_impurity_drop = len(indices) / len(y) * (node.impurity - lowest_impurity) >= self.min_impurity_decrease

        # if we manage to find a decent split
        if big_impurity_drop:
            ## record feature importance
            self.feature_importances_[node.fidx] += len(indices) / len(y) * (node.impurity - lowest_impurity)

            node.left = _Node(depth=node.depth+1)
            node.right = _Node(depth=node.depth+1)
            self.n_leaf_nodes += 1
            self._build_tree(node.left, X, y, best_left_indices)
            self._build_tree(node.right, X, y, best_right_indices)
        else:                                                               # leaf node
            node.samples = len(indices)
            class_codes, counts = np.unique(y[indices], return_counts=True)
            value = np.zeros(self.classes_.shape, dtype=int)
            value[class_codes] = counts                                     # essentially pads with zeros all the currently nonexistent classes
            node.value = value
            node.class_ = self.classes_[class_codes[counts.argmax()]]

    def fit(self, X, y):
        '''
        Creates the tree and sets the learned attributes.
        '''
        # setting some of the learned attributes
        self.classes_, y = np.unique(y, return_inverse=True) # classes_ in sorted order; y's values are now encoded as 0, 1, 2, etc.
        self.n_classes_ = len(self.classes_)
        self.n_features_in_ = X.shape[1]
        self.feature_importances_ = np.zeros((self.n_features_in_, ))

        # configuring the float-valued hyperparameters to make them integers
        if type(self.min_samples_split) is float:
            self.min_samples_split = round(self.min_samples_split * len(y))
        if type(self.min_samples_leaf) is float:
            self.min_samples_leaf = round(self.min_samples_leaf * len(y))
        if type(self.max_features) is int:
            pass
        elif type(self.max_features) is float:
            self.max_features = max(1, int(self.max_features * self.n_features_in_))
        elif self.max_features == "sqrt":
            self.max_features = round(np.sqrt(self.n_features_in_))
        elif self.max_features == "log2":
            self.max_features = round(np.log2(self.n_features_in_))
        elif self.max_features is None:
            self.max_features = self.n_features_in_
        else:
            raise ValueError("Unrecognized value for 'max_features': " + self.max_features)
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # more configs
        self.n_leaf_nodes = 1

        # actually creating the tree
        self.tree_ = _Node(depth=0)
        indices = np.arange(X.shape[0])
        self._build_tree(self.tree_, X, y, indices)
        self._fitted = True
        return self
    
    def predict_proba(self, X):
        if not self._fitted:
            raise sklearn.exceptions.NotFittedError("MyTreeClassifer instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
        y = [None] * X.shape[0]
        for i in X.shape[0]: # rows / instances
            x = X[i]
            node = self.tree_
            while node.left is not None and node.right is not None:
                node = node.left if x[node.fidx] <= node.threshold else node.right
            y[i] = node.value / node.samples
        return np.array(y)
            

    def predict(self, X):
        if not self._fitted:
            raise sklearn.exceptions.NotFittedError("MyTreeClassifer instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
    
    def score(self, X, y):
        return np.mean(y == self.predict(X)) #accuracy

def main():
    tree_clf = MyTreeClassifier()
    X = np.array([[0],
                  [1]])
    y = np.array(["A", "B"])
    tree_clf.fit(X, y)
    print("done")

if __name__ == "__main__":
    main()