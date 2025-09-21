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

### AREAS of IMPROVEMENT?
### Currently, the l1 regularization select features, but it only drops even-degree features (d >= 6)
### The l2 regularization is also too strong for the same alpha value, compared to sklearn's.
### TODO: implement partial_fit()


# Helper class: regularized cost function
class RegularizedCostFunc:
    def __init__(self, plain_gradient_func, l1_ratio, alpha):
        '''`plain_gradient_func` takes `theta`, `X`, `y`.'''
        self.plain_gradient_func = plain_gradient_func
        self.l1_ratio = l1_ratio
        self.alpha = alpha
    
    # def gradient(self, theta, X, y):
    #     '''The shape of the returned array is (n, 1).'''
    #     regularization_term = (1 - self.l1_ratio) * self.alpha * theta \
    #                             + self.l1_ratio * self.alpha * ((theta > 0).astype(np.float64) - (theta < 0).astype(np.float64)) #sgn(theta)
    #     regularization_term[0] = 0 # must not regularize the bias.
    #     return self.plain_gradient_func(theta, X, y) + regularization_term
    
    def update_theta(self, theta, X, y, eta):
        '''
        Returns a new theta that is equals to theta - eta * (gradient of regularized cost function).
        For l1 regularization, feature selection is performed after the l1 term is subtracted from theta. (If theta changes sign as a result of the update up to that point, it is set to zero.)
        For elasticnet regularization, the unregularized cost function gradient is first applied. Next, the l1 regularization gradient is applied. Then we perform feature selection. Finally, we apply the l2 regularization term. This allows the l2 regularization to mediate the feature selection.
        '''
        new_theta = theta - eta * self.plain_gradient_func(theta, X, y)

        if self.l1_ratio > 0:
            # l1 regularization
            l1_reg_term = self.l1_ratio * self.alpha * ((theta > 0).astype(np.float64) - (theta < 0).astype(np.float64)) #sgn(theta)
            l1_reg_term[0] = 0
            new_theta -= eta * l1_reg_term

            # do this feature selection only if we are doing lasso (l1_ratio == 1.0)
            new_theta[new_theta * theta < 0] = 0 # those that changed signs are set to zero

        # l2 regularization
        l2_reg_term = (1 - self.l1_ratio) * self.alpha * theta 
        l2_reg_term[0] = 0
        new_theta -= eta * l2_reg_term

        return new_theta


# Implement minibatch gradient descent.
# Started simple. Then, added a learning schedule. Then added early stopping. Then added a learning curve by iterations. Then, I added a standard scaling before training.
# Next, I refactored it in the style of a sklearn estimator
# Afterwards, I added a regularization to the cost function

class MinibatchGDRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, n_epoch=100, minibatch_ratio=0.05, 
                 eta0=0.1, power_t=0.1, 
                 validation_fraction=0.2, patience=50, tol=1e-5,
                 penalty=None, alpha=1.0, l1_ratio=0.1,
                 random_state=None):
        '''`penalty` is one of `l2`, `l1`, `elasticnet`, `None`.'''
        super().__init__()
        self.n_epoch = n_epoch
        self.minibatch_ratio = minibatch_ratio
        self.n_minibatch = round(1 / minibatch_ratio)
        self.eta0 = eta0
        self.power_t = power_t
        self.validation_fraction = validation_fraction
        self.patience = patience
        self.tol = tol
        self.random_state = random_state

        # regularization parameters
        if penalty is None:
            alpha = 0.0
        elif penalty == "l2":
            l1_ratio = 0.0
        elif penalty == "l1":
            l1_ratio = 1.0
        elif penalty == "elasticnet":
            pass
        else:
            raise ValueError("Penalty must be one of 'l2', 'l1', 'elasticnet', or None.")
        self.penalty = penalty
        self.regularized_cost_func = RegularizedCostFunc(
            plain_gradient_func = lambda theta, X, y: 2 / X.shape[0] * X.T @ (X @ theta - y),
            l1_ratio = l1_ratio,
            alpha = alpha
        )

        self._is_fitted = False
        
    
    def fit(self, X, y):
        # add dummy feature
        m = X.shape[0]
        X_b = np.c_[np.ones((m, )), X]
        n = X_b.shape[1] # in fit()

        # initialize theta
        if self.random_state is not None:
            np.random.seed(self.random_state)
        theta = np.random.rand(n, 1)

        # learning schedule
        def invscaling(t):
            return self.eta0 / (t ** self.power_t)
        
        # splitting off a validation portion (for early stopping). We define minibatch_size here!
        shuffled_indices = np.random.permutation(m)
        X_b_shuffled = X_b[shuffled_indices]
        y_shuffled = y[shuffled_indices]

        vs = round(self.validation_fraction * m) #validation size
        X_train_b = X_b_shuffled[vs : ]
        y_train = y_shuffled[vs : ]
        X_val_b = X_b_shuffled[ : vs]
        y_val = y_shuffled[ : vs]
        minibatch_size = round(vs * self.minibatch_ratio)

        # preparing for early stopping
        best_err = np.inf
        self.theta_ = None # best theta
        pcount = 0

        # preparing a makeshift learning curve
        self.terr_list = []
        self.verr_list = []

        for epoch in range(self.n_epoch):
            # shuffle the training data (X_train_b, y_train together)
            shuffled_indices = np.random.permutation(m - vs)
            X_train_b_shuffled = X_train_b[shuffled_indices]
            y_train_shuffled = y_train[shuffled_indices]

            for i in range(self.n_minibatch):
                # select minibatches in order
                X_minibatch = X_train_b_shuffled[i * minibatch_size : (i+1) * minibatch_size]
                y_minibatch = y_train_shuffled[i * minibatch_size : (i+1) * minibatch_size]

                # main step of the algorithm
                # gradient = self.regularized_cost_func.gradient(theta, X_minibatch, y_minibatch) #2 / minibatch_size * X_minibatch.T @ (X_minibatch @ theta - y_minibatch)
                # theta = theta - invscaling(epoch * m + (i + 1) * minibatch_size) * gradient
                theta = self.regularized_cost_func.update_theta(theta, X_minibatch, y_minibatch, invscaling(epoch * m + (i + 1) * minibatch_size))

            # early stopping
            val_err = np.mean((X_val_b @ theta - y_val)**2)

                # recording for our makeshift learning curve
            self.verr_list.append(val_err)
            self.terr_list.append(np.mean((X_train_b @ theta - y_train) ** 2))

            if val_err > best_err - self.tol:
                pcount += 1
            else:
                pcount = 0
            if pcount >= self.patience:
                break
            if val_err < best_err:
                best_err = val_err
                self.theta_ = theta

        self._is_fitted = True
        self.n_iter_ = epoch - self.patience + 1
        self.intercept_ = self.theta_[0]
        self.coef_ = self.theta_[1: ].flatten()
        return self
        

    def predict(self, X):
        if not self._is_fitted:
            raise sklearn.exceptions.NotFittedError("This MinibatchGDRegressor instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b @ self.theta_

    def display_learning_curve(self):
        if not self._is_fitted:
            raise sklearn.exceptions.NotFittedError("This MinibatchGDRegressor instance is not fitted yet. Call 'fit' with appropriate arguments before viewing the training results!")
        terr_list = np.array(self.terr_list)
        verr_list = np.array(self.verr_list)
        iterations_list = np.arange(len(terr_list))
        plt.figure(figsize=(6, 4))
        plt.plot(iterations_list, terr_list, "r-+", label="train")
        plt.plot(iterations_list, verr_list, "b-", label="validation")
        plt.axvline(x = self.n_iter_, linestyle="--", label="Early stopping.")
        plt.xlabel("Iterations")
        plt.ylabel("MSE", rotation="horizontal", labelpad=20)
        plt.title("Learning Curve")
        plt.grid()
        plt.legend()
        plt.show()


def learning_curve_by_iterations(estimator, X, y, *, display=False, max_iter=1000, error_fn=None,
                                 early_stopping=True, validation_fraction=0.33, patience=20, tol=1e-5):
    '''`estimator` must implement `partial_fit()` unless estimator is a pipeline, and must `fit_intercept`.\n
    If `estimator` is a pipeline, we assume that the transformers only modify `X` and not `y`.\n
    `estimator` outputs should be single target.\n
    Uses `mean_squared_error` for regressors and `accuracy` for classifiers unless specified otherwise.\n
    Returns: training_iterations, training_errors, validation_errors'''
    # split the data into training and validation (shuffle first)
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of instances.")
    m = X.shape[0]
    np.random.seed(45)
    shuffled_indices = np.random.permutation(m)
    X_shuffled = X[shuffled_indices]
    y_shuffled = y[shuffled_indices]
    vs = round(validation_fraction * m) #validation size
    X_val = X_shuffled[ : vs]
    y_val = y_shuffled[ : vs]
    X_train = X_shuffled[vs : ]
    y_train = y_shuffled[vs : ]

    # setup
    if error_fn == None:
        if sklearn.base.is_regressor(estimator):
            error_fn = mean_squared_error
            predict_fn = lambda X: estimator.predict(X)
        elif sklearn.base.is_classifier(estimator):
            error_fn = log_loss
            predict_fn = lambda X: estimator.predict_proba(X)
        else:
            raise ValueError("Estimator is neither a classifier nor a regressor (based on the 'estimator_type' tag)")
    if hasattr(estimator, "partial_fit"):
        pass
    elif isinstance(estimator, sklearn.pipeline.Pipeline) and hasattr(estimator[-1], "partial_fit"):
        def partial_fit(self, X, y):
            for transformer in estimator[ : -1]:
                X = transformer.fit_transform(X)
            estimator[-1].partial_fit(X, y)
            return self
        estimator.partial_fit = types.MethodType(partial_fit, estimator)
    elif "warm_start" in estimator.get_params():
        estimator.set_params(warm_start=True, max_iter=1)
    elif "warm_start" in estimator[-1].get_params(): # is a pipeline
        estimator[-1].set_params(warm_start=True, max_iter=1)
    else:
        raise ValueError("Estimator must implement the 'partial_fit' method or have a 'warm_start' parameter in its constructor in order for this function to work.")
    
    # init variables
    terr_list = []
    verr_list = []
    best_error = np.inf
    count = 0
    early_stopped = False

    for epoch in range(max_iter):
        # update weights several times
        try:
            estimator.partial_fit(X_train, y_train.ravel())
        except AttributeError:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                estimator.fit(X_train, y_train.ravel()) #warm_start==True

        # compute errors and record them
        train_error = error_fn(y_train.ravel(), predict_fn(X_train))
        val_error = error_fn(y_val.ravel(), predict_fn(X_val))

        terr_list.append(train_error)
        verr_list.append(val_error)

        # perform early stopping if specified
        if early_stopping:
            if val_error > best_error - tol:
                count += 1
            else:
                 count = 0
            if count >= patience:
                early_stopped = True
                break
            best_error = min(val_error, best_error)

    # displaying the learning curve
    if display:
        iterations_list = np.arange(len(terr_list))
        plt.figure(figsize=(6, 4))
        plt.plot(iterations_list, terr_list, "r-+", label="Train")
        plt.plot(iterations_list, verr_list, "b-", label="Validation")
        if early_stopped:
            plt.axvline(x = epoch - patience, linestyle="--", label="Early stopping.")
        plt.xlabel("Iterations")
        if hasattr(error_fn, "__name__"):
            plt.ylabel(error_fn.__name__, rotation="horizontal", labelpad=20)
        plt.title("Learning Curve")
        plt.grid()
        plt.legend()
        plt.show()

    return np.arange(epoch + 1), np.array(terr_list), np.array(verr_list)

# First, we lay out the class architecture and write a simple version with just minibatch.
# Next, we add a learning schedule, early stopping, regularization, partial_fit() for use with learning_curve_by_iterations

class SoftmaxRegressor(ClassifierMixin, BaseEstimator):
    def __init__(self, n_epoch=100, minibatch_ratio=0.05, 
                 eta0=0.1, power_t = 0.1,
                 early_stopping=True, validation_fraction = 0.2, patience=20, tol=1e-5,
                 penalty=None, alpha=1.0, l1_ratio=0.1,
                 warm_start = False, random_state=None):
        '''
        A softmax regressor (generalized logistic regression), trained by mini-batch gradient descent.\n
        The learned attribute `coef_` follows the format of sklearn, having the shape (n_classes, n_features). \n
        The attribute `Theta_` and the variable `Theta` have shape (n_features + 1 , n_classes), since this is more useful for computations.\n
        The parameter `penalty` must have one of the values `l2`, `l1`, `elasticnet`, `None`.
        '''
        self.n_epoch = n_epoch
        self.minibatch_ratio = minibatch_ratio

        self.eta0 = eta0
        self.power_t = power_t

        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.patience = patience
        self.tol = tol

        self.penalty = penalty
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        if self.penalty is None:
            self.alpha = 0
            self.l1_ratio = 0
        elif self.penalty == "l2":
            self.l1_ratio = 0
        elif self.penalty == "l1":
            self.l1_ratio = 1
        elif self.penalty == "elasticnet":
            pass
        else:
            raise ValueError("Penalty must be one of 'l2', 'l1', 'elasticnet', or None.")
        
        self.warm_start = warm_start
        self.random_state = random_state
        self._is_fitted = False

        # setting up the regularization cost function
        self.regularized_cost_func = RegularizedCostFunc(
            plain_gradient_func=lambda Theta, X, Y: 1 / X.shape[0] * X.T @ (SoftmaxRegressor._sigma(X, Theta) - Y),
            alpha=self.alpha,
            l1_ratio=self.l1_ratio
        )
    
    @staticmethod
    def _sigma(X, Theta):
        '''The scoring plus softmax function rolled into one. A helper function.\n
        returned.shape = (n_instances, n_classes)'''
        tmp = np.exp(X @ Theta)
        return tmp / np.sum(tmp, axis=1, keepdims=True)
    
    def _invscaling(self, t):
        '''invscaling learning schedule. `t` is the number of instances seen.'''
        return self.eta0 / (t ** self.power_t)
    
    def fit(self, X, y):
        '''
        Takes training data `X`, `y` and continues to train the model on the data (for one epoch, using the coefficients the model already has, if it has any).\n
        Returns the fitted estimator.\n
        '''
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of instances.")
        
        # add dummy feature
        X = np.c_[np.ones((X.shape[0], 1)), X]

        # one-hot encode y to turn it into Y. store the classes as self.classes_
        self.classes_, y = np.unique(y, return_inverse=True)
        Y = np.stack([y == classcode for classcode in range(self.classes_.shape[0])], 
                     axis=1).astype(int)                                                # this `axis=1` means that y can be a column or a row. It doesn't matter.
        
        # shuffle the data
        if self.random_state is not None:
            np.random.rand(self.random_state)
        shuffled_indices = np.random.permutation(X.shape[0])
        X_shuffled = X[shuffled_indices]
        Y_shuffled = Y[shuffled_indices]

        # train-validation split for early stopping
        if self.early_stopping:
            vs = round(self.validation_fraction * X.shape[0]) # validation size
            X_train = X_shuffled[vs : ]
            Y_train = Y_shuffled[vs : ]
            X_val = X_shuffled[ : vs]
            Y_val = Y_shuffled[ : vs]
        else:
            X_train = X_shuffled
            Y_train = Y_shuffled
            X_val = None
            Y_val = None

        # initialize Theta
        if self.warm_start and self._is_fitted:
            Theta = self.Theta_
        else:
            if self.random_state is not None:
                np.random.rand(self.random_state)
            Theta = np.random.rand(X_train.shape[1], self.classes_.shape[0])

        # preparing for early stopping
        best_loss = np.inf
        count = 0
        early_stopped = False

        # minibatch gradient descent
        for epoch in range(self.n_epoch):
            # shuffle the data
            shuffled_indices = np.random.permutation(X_train.shape[0])
            X_train_shuffled = X_train[shuffled_indices]
            Y_train_shuffled = Y_train[shuffled_indices]

            # split the data for each minibatch
            n_minibatch = round(1 / self.minibatch_ratio)
            minibatch_size = round(self.minibatch_ratio * X_train.shape[0])
            for i in range(n_minibatch):
                X_minibatch = X_train_shuffled[i * minibatch_size : (i + 1) * minibatch_size]
                Y_minibatch = Y_train_shuffled[i * minibatch_size : (i + 1) * minibatch_size]

                # gradient descent
                # gradient = self.regularized_cost_func.gradient(Theta, X_minibatch, Y_minibatch) # 1 / minibatch_size * X_minibatch.T @ (SoftmaxRegressor._sigma(X_minibatch, Theta) - Y_minibatch)
                # Theta = Theta - self.invscaling(epoch * X_train.shape[0] + (i + 1) * minibatch_size) * gradient
                Theta = self.regularized_cost_func.update_theta(Theta, X_minibatch, Y_minibatch, self._invscaling(epoch * X_train.shape[0] + (i + 1) * minibatch_size))
            
            # the early stopping part
            if self.early_stopping:
                # compute validation log loss
                val_loss = (-1) / X_val.shape[0] * np.dot(Y_val.ravel(), SoftmaxRegressor._sigma(X_val, Theta).ravel())

                # decide whether to do an early stopping
                if val_loss > best_loss - self.tol:
                    count += 1
                else:
                    count = 0
                if val_loss < best_loss:
                    best_loss = val_loss
                    self.Theta_ = Theta # guaranteed to run at least once since best_loss is initialized as np.inf
                if count >= self.patience:
                    early_stopped = True
                    break

        if not self.early_stopping:
            self.Theta_ = Theta
        
        # setting the learned attributes (self.classes_ has already been set earlier.)
        self._is_fitted = True
        self.intercept_ = self.Theta_[0]
        self.coef_ = self.Theta_[1:].T
        self.n_features_in_ = self.coef_.shape[1]
        self.n_iter_ = epoch + 1 - self.patience if early_stopped else self.n_epoch

        return self
    
    def partial_fit(self, X, y):
        '''Takes training data `X`, `y` and continues to train the model on the data (for one epoch, using the coefficients the model already has, if it has any).\n
        Returns the fitted estimator.'''
        # We implement `partial_fit`` in terms of `fit` by setting warm_start=True and n_epoch=1 temporarily.
        tmp_warm_start = self.warm_start
        tmp_n_epoch = self.n_epoch
        self.warm_start = True
        self.n_epoch = 1
        self.fit(X, y)
        self.warm_start = tmp_warm_start
        self.n_epoch = tmp_n_epoch
        return self

    def predict_proba(self, X):
        '''returned.shape = (n_instances, n_classes)'''
        if not self._is_fitted:
            raise sklearn.exceptions.NotFittedError("This SoftmaxRegressor instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
        X = np.c_[np.ones((X.shape[0], 1)), X] # add dummy feature
        return SoftmaxRegressor._sigma(X, self.Theta_)
    

    def predict(self, X):
        '''returned.shape = (n_instances, )'''
        if not self._is_fitted:
            raise sklearn.exceptions.NotFittedError("This SoftmaxRegressor instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
        
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
    

#=====================TESTING========================================
def __test_mini_batch():
    # Play data 1: Linear
    np.random.seed(42)
    m = 1000  # number of instances
    X_train = 2 * np.random.rand(m, 1)
    y_train = 4 + 3 * X_train + np.random.randn(m, 1)
    X_test = 2 * np.random.rand(m, 1)
    y_test = 4 + 3 * X_test + np.random.randn(m, 1)

    mb_reg = MinibatchGDRegressor()
    mb_reg.fit(X_train, y_train)
    print("training error: ", np.mean((mb_reg.predict(X_train) - y_train) ** 2))
    print("test error: ", np.mean((mb_reg.predict(X_test) - y_test) ** 2))
    mb_reg.display_learning_curve()

def __test_learning_curve_by_iterations():
    # test data: quadratic
    np.random.seed(42)
    m = 1000
    X_train = 3 - 6 * np.random.rand(m, 1)
    y_train = 0.5 * X_train ** 2 + X_train + 2 + np.random.randn(m, 1)
    X_val = 3 - 6 * np.random.rand(m, 1)
    y_val = 0.5 * X_val ** 2 + X_val + 2 + np.random.randn(m, 1)

    poly_reg = make_pipeline(PolynomialFeatures(degree=10, include_bias=False),
                             StandardScaler(),
                             SGDRegressor(penalty=None, random_state=40))
    train_iters, train_errors, val_errors = learning_curve_by_iterations(poly_reg, X_train, y_train, display=True)
    print("training MSE:", mean_squared_error(y_train.ravel(), poly_reg.predict(X_train)))
    print("validation MSE:", mean_squared_error(y_val.ravel(), poly_reg.predict(X_val)))

def __test_regularized_minibatch_gd():
    # test data: quadratic
    np.random.seed(42)
    m = 1000
    X_train = 3 - 6 * np.random.rand(m, 1)
    y_train = 0.5 * X_train ** 2 + X_train + 2 + np.random.randn(m, 1)
    X_val = 3 - 6 * np.random.rand(m, 1)
    y_val = 0.5 * X_val ** 2 + X_val + 2 + np.random.randn(m, 1)
    
    my_list = [MinibatchGDRegressor(penalty="l2", alpha=1.0, random_state=42), 
               MinibatchGDRegressor(penalty="l1", alpha=0.1, random_state=42), 
               MinibatchGDRegressor(penalty="elasticnet", alpha=1.0, random_state=42)]
    sk_list = [Ridge(alpha=1.0, random_state=42), 
               Lasso(alpha=0.1, random_state=42), 
               ElasticNet(alpha=1.0, l1_ratio=0.1, random_state=42)]
    estimator_list = itertools.chain.from_iterable(zip(my_list, sk_list))

    for estimator in estimator_list:
        poly_reg = make_pipeline(PolynomialFeatures(degree=20, include_bias=False),
                                StandardScaler(),
                                estimator)
        poly_reg.fit(X_train, y_train)
        print(poly_reg[2].__class__.__name__, end="")
        if hasattr(poly_reg[2], "penalty"):
            print(" with penalty", poly_reg[2].penalty)
        else:
            print()
        print("training MSE: ", mean_squared_error(y_train, poly_reg.predict(X_train)))
        print("validation MSE: ", mean_squared_error(y_val, poly_reg.predict(X_val)))
        print("intercept and coefficients: \n", poly_reg[2].intercept_, *poly_reg[2].coef_)
        print()

def __test_softmax_no_regularization():
    # data: iris. Feature names: ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    X, y = sklearn.datasets.load_iris(return_X_y=True)
    X = X[ : , 2:]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # without regularization:
    # my regressor vs sklearn's
    print("Without regularization: \n")
    elist = [SoftmaxRegressor(random_state=47, minibatch_ratio=0.1, n_epoch=1000), 
             LogisticRegression(penalty=None, solver="saga", random_state=42, max_iter=5000)]
    for estimator in elist:
        pipe = make_pipeline(StandardScaler(), estimator)
        pipe.fit(X_train, y_train)
        print(estimator.__class__.__name__)
        print(f"Model trained for {estimator.n_iter_} epochs.")
        print("Training log_loss:", log_loss(y_train, pipe.predict_proba(X_train)))
        print("Test log_loss:", log_loss(y_test, pipe.predict_proba(X_test)))
        print("Model intercept: \n", estimator.intercept_)
        print("Model coef:\n", estimator.coef_)
        print()
    
    for estimator in elist:
        pipe = make_pipeline(StandardScaler(), estimator)
        learning_curve_by_iterations(pipe, X, y, max_iter=5000, display=True)

def __test_softmax_with_regularization():
    # data: iris. Feature names: ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    X, y = sklearn.datasets.load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # with regularization:
    # my regressor vs sklearn's
    print("With regularization: \n")
    elist = [SoftmaxRegressor(random_state=47, minibatch_ratio=0.1, n_epoch=1000, penalty="elasticnet", alpha=1), 
             LogisticRegression(penalty="elasticnet", C=0.015, l1_ratio=0.7, solver="saga", random_state=43, max_iter=5000)]
    
    for estimator in elist[1:]:
        pipe = make_pipeline(PolynomialFeatures(degree=4, include_bias=False), StandardScaler(), estimator)
        learning_curve_by_iterations(pipe, X, y, max_iter=5000, display=True)


def main():
    #__test_mini_batch()
    #__test_learning_curve_by_iterations()
    #__test_regularized_minibatch_gd()
    #__test_softmax_no_regularization()
    __test_softmax_with_regularization()

if __name__ == "__main__":
    main()