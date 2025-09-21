from abc import ABC, abstractmethod
from collections import deque

import numpy as np
import sklearn
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.inspection import DecisionBoundaryDisplay
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

def _Gini(y):
    '''Gini impurity of a categorical array'''

    _, counts = np.unique(y, return_counts=True)
    return 1 - np.sum((counts / counts.sum()) ** 2)

def _Entropy(y):
    _, counts = np.unique(y, return_counts=True)
    # - sigma p log p
    proportions = counts / counts.sum()
    return -sum(proportions * np.log2(proportions))

def _MSE(y):
    return np.var(y)

# DEVELOPMENT PROCESS 
# 1. Layout the structure of MyTreeClassifier, __Node and the training algorithm
# 2. start with a simple version of fit
# 3. added the entropy impurity measure
# 4. added in all the other hyperparameters (a lot of work)
# 5. decided that build tree should be a method of MyTreeClassifier (it is still recursive on nodes)
# 6. added in the feature_importances_ attribute
# 7. refactoring an abstact base class, adding a regressor class
# 8. fixing a bug with the max_leaf_nodes condition
# 9. to make the classes compatible with RandomizedSearchCV, we need to follow the convention of get_params(): init params = attributes
# 10. rewrote the build tree (heart of the CART algorithm), changing it from depth-first to breadth-first construction of the tree. KEY: this allows the max_leaf_nodes hyperparameter to work properly.

class AbstractTreeEstimator(BaseEstimator, ABC):
    '''
    Abstract base class for the CART algorithm
    '''
    def __init__(self,
                 criterion,
                 splitter="best",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 max_features=None,
                 random_state=None,
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.0):
        # should only store and not check their validities
        # the params could be altered by setattr or set_params
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease

        self._fitted = False


    def _build_tree(self, X, y):
        '''
        Takes a `node` and builds the decision tree with that as the root. returns None.\n
        `self` is used as a storage for all the regularization hyperparameters
        '''
        # breadth-first construction
        q = deque()
        q.append((self.tree_, np.arange(len(y))))
        while len(q) > 0:
            node, indices = q.popleft()
            
            # setup
            lowest_impurity = np.inf
            best_left_indices = None
            best_right_indices = None
            split_found = False
            big_impurity_drop = False
            node.impurity = self.impurity_measure(y[indices])
            
            # go condition
            if len(indices) >= self.min_samples_split and node.depth < self.max_depth and self.n_leaf_nodes < self.max_leaf_nodes:
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
                q.append((node.left, best_left_indices))
                q.append((node.right, best_right_indices))
            else:                                                               # leaf node
                node.samples = len(indices)
                self._handle_leaf_nodes(node, y, indices)

    @abstractmethod
    def _handle_leaf_nodes(self, node, y, indices):
        pass

    def fit(self, X, y):
        '''
        Creates the tree and sets the learned attributes.
        '''
        # setting some of the learned attributes
        self.n_features_in_ = X.shape[1]
        self.feature_importances_ = np.zeros((self.n_features_in_, ))

        # checking the validity of the parameters
        if self.splitter.lower() == "best":
            self.use_rand_thresh = False
        elif self.splitter.lower() == "random":
            self.use_rand_thresh = True  # is a extra randomized tree
        else:
            raise ValueError("Unrecognized splitter value: " + self.splitter)
        self.max_depth = self.max_depth if self.max_depth is not None else np.inf
        self.max_leaf_nodes = self.max_leaf_nodes if self.max_leaf_nodes is not None else np.inf

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
        self._build_tree(X, y)
        self._fitted = True
        return self     
           
    @abstractmethod
    def predict(self,X):
        pass
    
    @abstractmethod
    def score(self, X, y):
        pass


class MyTreeClassifier(ClassifierMixin, AbstractTreeEstimator):
    '''
    Implements the CART algorithm for classification.
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
        super().__init__(criterion=criterion,
                        splitter=splitter,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        max_features=max_features,
                        random_state=random_state,
                        max_leaf_nodes=max_leaf_nodes,
                        min_impurity_decrease=min_impurity_decrease)

    def _handle_leaf_nodes(self, node, y, indices):
        class_codes, counts = np.unique(y[indices], return_counts=True)
        value = np.zeros(self.classes_.shape, dtype=int)
        value[class_codes] = counts                                     # essentially pads with zeros all the currently nonexistent classes
        node.value = value
        
    
    def fit(self, X, y):
        # setting classifier-specific attributes
        self.classes_, y = np.unique(y, return_inverse=True) # classes_ in sorted order; y's values are now encoded as 0, 1, 2, etc.
        self.n_classes_ = len(self.classes_)

        # checking the validity of the criterion
        if self.criterion.lower() == "gini":
            self.impurity_measure = _Gini
        elif self.criterion.lower() == "entropy":
            self.impurity_measure = _Entropy
        else:
            raise ValueError("Unrecognized criterion: " + self.criterion)
        return super().fit(X, y)
    

    def predict_proba(self, X):
        if not self._fitted:
            raise sklearn.exceptions.NotFittedError("MyTreeClassifer instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
        y = [None] * X.shape[0]
        for i in range(X.shape[0]): # rows / instances
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

class MyTreeRegressor(RegressorMixin, AbstractTreeEstimator):
    '''
    Implements the CART algorithm for regression.
    '''
    def __init__(self, 
                 criterion="squarred_error",
                 splitter="best",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 max_features=None,
                 random_state=None,
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.0):
        '''
        `criterion`: {"squared_error"}
        `splitter`: {"best", "random"} 
        '''
        super().__init__(criterion=criterion,
                        splitter=splitter,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        max_features=max_features,
                        random_state=random_state,
                        max_leaf_nodes=max_leaf_nodes,
                        min_impurity_decrease=min_impurity_decrease)

    def fit(self, X, y):
        # checking the validity of the criterion
        if self.criterion.lower() == "squared_error":
            self.impurity_measure = _MSE
        else:
            raise ValueError("Unrecognized criterion: " + self.criterion)
        return super().fit(X, y)
    
    def _handle_leaf_nodes(self, node, y, indices):
        node.value = y[indices].mean()

    def predict(self, X):
        y = [None] * X.shape[0]
        node = self.tree_
        for i in range(X.shape[0]):
            x = X[i]
            while node.left is not None and node.right is not None:
                node = node.left if x[node.fidx] <= node.threshold else node.right
            y[i] = node.value
        return np.array(y)

    def score(self, X, y): #r^2 score
        y_pred = self.predict(X)
        u = np.sum((y - y_pred) ** 2)
        v = np.sum((y - y.mean()) ** 2)
        return 1 - u / v
        

def _test_baby_tree():
    # baby tree
    tree_clf = MyTreeClassifier()
    X = np.array([[0],
                  [1]])
    y = np.array(["A", "B"])
    tree_clf.fit(X, y)
    print()

def _test_decision_boundary():
    # prepare some moon data
    X, y = sklearn.datasets.make_moons(n_samples=10000, noise=0.4, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)
    # # create a model (no StandardScaler and Pipeline needed, no PolynomialFeatures needed either)
    my_tree_clf = MyTreeClassifier(max_depth=2, max_leaf_nodes=53, min_samples_split=2, random_state=41)
    # tree_clf = DecisionTreeClassifier(max_depth=7, max_leaf_nodes=28, min_samples_split=5, random_state=41)
    my_tree_clf.fit(X_train, y_train)
    # tree_clf.fit(X_train, y_train)
    print("My tree classifier's test accuracy:", my_tree_clf.score(X_test, y_test))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="inferno")
    disp = DecisionBoundaryDisplay.from_estimator(my_tree_clf, X, response_method="predict", ax=plt.gca(), alpha=0.5, cmap="viridis_r")
    plt.show()

def _test_RSCV():
    # prepare some moon data
    X, y = sklearn.datasets.make_moons(n_samples=10000, noise=0.4, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)   

    # randomized search cv
    my_tree_clf = MyTreeClassifier(random_state=41)
    param_distrib = {
        "max_leaf_nodes": np.arange(2, 100),
        "max_depth": np.arange(2, 10),
        "min_samples_split": np.arange(2, 6)
    }
    rscv = RandomizedSearchCV(my_tree_clf, param_distrib, n_iter=10, cv=3)
    rscv.fit(X_train, y_train)
    print("Best estimator: ", repr(rscv.best_estimator_))
    print("Mean val accuracy score of the best estimator:", rscv.best_score_)
    print("Test accuracy: ", rscv.best_estimator_.score(X_test, y_test))


def main():
    #_test_baby_tree()
    #_test_decision_boundary()
    _test_RSCV()

    print("done")

if __name__ == "__main__":
    main()