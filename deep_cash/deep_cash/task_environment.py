"""A handler for generating tasks and datasets and evaluating ml frameworks."""

import numpy as np
import pynisher
import warnings

from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

FIT_GRACE_PERIOD = 30


def _metafeatures(X_train, y_train):
    # For now this just creates a single metafeature:
    # number training examples
    return np.array([len(X_train)])


def _ml_framework_fitter(ml_framework, X, y):
    """Fits an ML framework to training data.

    This function handles warnings and errors

    :returns: a valid ml_framework if it successfully fits a model, None if
        calling the `fit` method leads to
    """
    # TODO: log these warnings/errors
    # raise numpy overflow errors
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with np.errstate(over="raise"):
            try:
                ml_framework.fit(X, y)
                return ml_framework
            except:
                return None


def _ml_framework_predict(ml_framework, X):
    """Generates predictions from a fit ml_framework.

    Handles errors at the predict layer, any over which need to be handled
    by the evaluation function.
    """
    with np.errstate(over="raise"):
        try:
            if hasattr(ml_framework, "predict_proba"):
                return ml_framework.predict_proba(X)[:, 1]
            return ml_framework.predict(X)
        except:
            return None


class TaskEnvironment(object):
    """Generates datasets associated with supervised learning tasks."""

    def __init__(
            self, scorer, n_samples=1000, random_state=None,
            per_framework_time_limit=10, per_framework_memory_limit=3072):
        self.scorer = scorer
        self.n_samples = n_samples
        self.random_state = random_state
        self.per_framework_time_limit = per_framework_time_limit
        self.per_framework_memory_limit = per_framework_memory_limit

        # TODO: soon this should be a set of datasets from which to sample.
        self.data_env = make_classification(
            n_samples=self.n_samples, n_features=50, n_informative=3,
            n_classes=2, shuffle=True, random_state=random_state)
        self.X = self.data_env[0]
        self.y = self.data_env[1]
        self.data_env_index = np.array(range(n_samples))
        self.n = len(self.data_env_index)
        np.random.seed(self.random_state)

        # enforce resource contraints on framework fitter
        self.ml_framework_fitter = pynisher.enforce_limits(
            mem_in_mb=self.per_framework_memory_limit,
            wall_time_in_s=self.per_framework_time_limit,
            grace_period_in_s=FIT_GRACE_PERIOD)(_ml_framework_fitter)

    def sample(self):
        """Samples the current data_env for features and targets.

        This function sets the instance variables:

        :ivar X_train: array of shape n_train x m where n_train is number of
            training samples and m is number of features.
        :ivar y_train: array of shape n_train x k where k is number of
            classes/targets.
        :ivar X_test: array of shape n_test x m where n_test is number of
            test samples.
        :ivar y_test: array of shape n_test x k.
        :returns: array of shape 1 x m_meta where m_meta is the number of
            metafeatures for a particular sample.
        """
        # number of training samples to bootstrap
        train_index = np.random.choice(
            self.data_env_index, self.n_samples, replace=True)
        test_index = np.setdiff1d(self.data_env_index, train_index)
        # save the test partitions for evaluation
        self.X_train = self.X[train_index]
        self.y_train = self.y[train_index]
        self.X_test = self.X[test_index]
        self.y_test = self.y[test_index]
        return _metafeatures(self.X_train, self.y_train)

    def evaluate(self, ml_framework):
        return self._fit_score(ml_framework)

    def _fit(self, ml_framework):
        # TODO: log these warnings/errors
        ml_framework = self.ml_framework_fitter(
            ml_framework, self.X_train, self.y_train)
        if self.ml_framework_fitter.exit_status != 0:
            return None
        return ml_framework

    def _score(self, ml_framework):
        try:
            pred = _ml_framework_predict(ml_framework, self.X_test)
            return self.scorer(self.y_test, pred) * 100  # scale to [0, 100]
        except:
            return None

    def _fit_score(self, ml_framework):
        ml_framework = self._fit(ml_framework)
        if ml_framework is None:
            return None
        return self._score(ml_framework)

