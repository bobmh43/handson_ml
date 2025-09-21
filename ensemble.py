import numpy as np
import sklearn
from sklearn.base import RegressorMixin, ClassifierMixin, BaseEstimator, clone
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
#from sklearn.utils import Bunch

# DEVELOPMENT PROCESS
# 1. Layout the interface of MyStackingClassifier
# 2. Implement the most basic version of fit, predict_proba, predict, ignoring all the init parameters
# 3. check the validity of the init params at the start of fitting
# 4. fix the stack_method thingy. (init parameter and during fitting.)
# 5. added the passthrough functionality
#TODO: fix this: estimators and named_estimators should vary together. Change one and you change the other. Currently, we removed self.named_estimators and self.named_estimators_

def MyStackingClassifier(ClassifierMixin, BaseEstimator):
    '''
    My implementation of the stacking ensemble, for classification. Note, this has a very slow training process.\n
    Big note: To create a stacking ensemble of three layers, first ignore the bottom layer. Make a StackingClassifier using the top two layers. Now, create another StackingClassifier using the bottom row and have the previous StackingClassifier be the final_estimator.\n
        example:
        ```
        layer1 = [("linearsvc", LinearSVC()), ("extra_trees", ExtraTreesClassifier()), ("mlp", MLPClassifier())]
        layer2 = [("forest", RandomForest()), ("gradient", GradientBoostingClassifier()))]
        layer3 = LogisticRegression()
        stack23 = StackingClassifier(layer2, layer3)
        stack123 = StackingClassifier(layer1, stack23)
        stack123.fit(X, y)
        ```
    '''
    def __init__(self, 
                 estimators, final_estimator=None, 
                 *, cv=None, n_jobs=None, stack_method='auto', passthrough=False):
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.cv = cv
        self.n_jobs = n_jobs
        self.stack_method = stack_method
        self.passthrough = passthrough

        self._fitted = False

    def fit(self, X, y):
        # check the validity of the init parameters
        self.final_estimator = self.final_estimator if self.final_estimator is not None else LogisticRegression()
        
        if self.stack_method not in {"auto", "predict_proba", "decision_function", "predict"}:
            raise ValueError("stack_method must be one of {'auto', 'predict_proba', 'decision_function', 'predict'}.")
        
        # actual operation
        return self._fit(X, y)
    
    def _fit(self, X, y):
        # set these learned attributes
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]

        # setup: decide which method to use for each estimator
        if self.stack_method != "auto":
            for name, estimator in self.estimators:
                if not hasattr(estimator, self.stack_method):
                    raise ValueError(f"Underlying estimator {name} does not implement the method '{self.stack_method}'.")
            self.stack_method_ = [self.stack_method] * len(self.estimators)
        else: # auto
            self.stack_method_ = []
            for name, estimator in self.estimators:
                if hasattr(estimator, "predict_proba"):
                    self.stack_method_.append("predict_proba")
                elif hasattr(estimator, "decision_function"):
                    self.stack_method_.append("decision_function")
                elif hasattr(estimator, "predict"):
                    self.stack_method_.append("predict")
                else:
                    raise ValueError(f"Underlying estimator {name} does not implement any of the methods 'predict_proba', 'decision_function', or 'predict'.") 




        # first, we train the first layer, each estimator uses the full set
        self.estimators_ = [clone(clf).fit(X, y) for _, clf in self.estimators]

        # next, pass X through the first layer by cross_val_predict (out-of-fold predictions)
            # note that cross_val_predict first makes an unfitted clone of clf.        
        predictions = (
            cross_val_predict(clf, X, y, cv=self.cv, n_jobs=self.n_jobs, method=method) 
            for clf, method in zip(self.estimators_, self.stack_method_)
        )
        all_2d = (
            arr.reshape((-1, 1)) if arr.ndim == 1 else arr 
            for arr in predictions
        )
        if len(self.classes_) == 2:                                                             # binary classification
            all_2d = (arr[:, 0] for arr in all_2d)
        X_new = np.hstack(list(all_2d))
        if self.passthrough:
            X_new = np.hstack((X, X_new))
        self.final_estimator_ = clone(self.final_estimator).fit(X_new, y)

        self._fitted = True
        return self

    def _fit_first(func): # decorator
        def inner(self, *args, **kwargs):
            if not self._fitted:
                raise sklearn.exceptions.NotFittedError("This MyStackingClassifier instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
            return func(self, *args, **kwargs)
        return inner
    
    def _map_X(self, X):
        # passing X through the bottom layer using whatever methods were used before during training
        predictions = (
            getattr(estimator, method)(X) 
            for estimator, method in zip(self.estimators_, self.stack_method_)
        )
        all_2d = (
            arr.reshape((-1, 1)) if arr.ndim == 1 else arr 
            for arr in predictions
        )
        if len(self.classes_) == 2:                                                             # binary classification
            all_2d = (arr[:, 0] for arr in all_2d)
        X_new = np.hstack(list(all_2d))
        if self.passthrough:
            X_new = np.hstack((X, X_new))
        return X_new

    @_fit_first
    def predict_proba(self, X):
        if not hasattr(self.final_estimator_, "predict_proba"):
            raise AttributeError("The final_estimator does not have the method 'predict_proba'.")  
        return self.final_estimator_.predict_proba(self._map_X(X)) 

    @_fit_first
    def predict(self, X):
        return self.final_estimator_.predict(self._map_X(X))

    @_fit_first
    def score(self, X, y, sample_weight=None): #accuracy
        if sample_weight is None:
            sample_weight = np.ones_like(y, dtype=np.float64)
        return sample_weight[self.predict(X) == y].sum() / sample_weight.sum()

    @_fit_first
    def transform(self, X):
        return self._map_X(X)

    @_fit_first
    def fit_transform(self, X, y): # i assume this is mostly for debugging? It does not make sense to use this otherwise, for fear of data leakage
        return self.fit(X, y).transform(X)


