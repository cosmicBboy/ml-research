"""A handler for generating tasks and datasets and evaluating ml frameworks."""

import numpy as np
import pynisher
import warnings

from sklearn.metrics import roc_auc_score

from .data_environments import classification_environments
from .errors import NoPredictMethodError, ExceededResourceLimitError


FIT_GRACE_PERIOD = 30

FIT_PREDICT_ERRORS = (
    FloatingPointError,
    TypeError,
    ValueError,
    RuntimeWarning,
    UserWarning)

NEGATIVE_REWARD_EXCEPTIONS = tuple(
    [e for e in FIT_PREDICT_ERRORS] +
    [NoPredictMethodError, ExceededResourceLimitError]
)


class TaskEnvironment(object):
    """Generates datasets associated with supervised learning tasks."""

    def __init__(
            self, scorer=roc_auc_score, random_state=None,
            per_framework_time_limit=10, per_framework_memory_limit=3072,
            error_reward=-1, reward_scale=100):
        """Initialize task environment."""
        # TODO: need to add an attribute that normalizes the output of the
        # scorer function to support different tasks. Currently, the default
        # is ROC AUC scorer on just classification tasks. Will need to
        # extend this for regression task error metrics.
        self.scorer = scorer
        self.random_state = random_state
        self.per_framework_time_limit = per_framework_time_limit
        self.per_framework_memory_limit = per_framework_memory_limit
        self.data_distribution = classification_environments.envs()
        self._error_reward = -1
        self._reward_scale = reward_scale

        # enforce resource contraints on framework fitter
        # TODO: support heuristic schedule for changing these contraints
        # over time.
        self.ml_framework_fitter = pynisher.enforce_limits(
            mem_in_mb=self.per_framework_memory_limit,
            wall_time_in_s=self.per_framework_time_limit,
            grace_period_in_s=FIT_GRACE_PERIOD)(_ml_framework_fitter)

        np.random.seed(self.random_state)

    @property
    def error_reward(self):
        """Return reward in the situation of a fit/predict error."""
        return self._error_reward * self._reward_scale

    @property
    def correct_hyperparameter_reward(self):
        """Return reward when proposed hyperparameter is correct."""
        return 1 * self._reward_scale

    def sample_data_env(self):
        """Sample the data distribution."""
        data_env_tuple = self.data_distribution[
            np.random.randint(len(self.data_distribution))]
        create_data_env, task_type, target_preprocessor = data_env_tuple
        data_env = create_data_env()
        self.X = data_env["data"]
        self.y = data_env["target"]
        self.task_type = task_type
        if target_preprocessor is not None:
            self.y = target_preprocessor().fit_transform(self.y.reshape(-1, 1))
        self.data_env_index = np.array(range(self.X.shape[0]))
        self.n_samples = len(self.data_env_index)

    def sample(self):
        """Sample the current data_env for features and targets.

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
        """Evaluate an ML framework by fitting and scoring on data."""
        try:
            return self._fit_score(ml_framework)
        except NEGATIVE_REWARD_EXCEPTIONS as e:
            return None

    def _fit(self, ml_framework):
        # TODO: log these warnings/errors
        ml_framework = self.ml_framework_fitter(
            ml_framework, self.X_train, self.y_train)
        if self.ml_framework_fitter.exit_status != 0:
            raise ExceededResourceLimitError(
                "ml framework %s exceeded with status: %s" % (
                    ml_framework, self.ml_framework_fitter.exit_status))
        return ml_framework

    def _score(self, ml_framework):
        pred = _ml_framework_predict(
            ml_framework, self.X_test, self.task_type)
        return self.scorer(self.y_test, pred) * self._reward_scale

    def _fit_score(self, ml_framework):
        return self._score(self._fit(ml_framework))


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
    # raise numpy overflow errors
    # TODO: log these warnings/errors
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with np.errstate(all="raise"):
            try:
                ml_framework.fit(X, y)
                return ml_framework
            except FIT_PREDICT_ERRORS:
                return None
            except Exception:
                raise


def _binary_predict_proba(pred_proba):
    return pred_proba[:, 1]


def _multiclass_predict_proba(pred_proba):
    return np.concatenate([p[:, [1]] for p in pred_proba], axis=1)


def _ml_framework_predict(ml_framework, X, task_type):
    """Generate predictions from a fit ml_framework.

    Handles errors at the predict layer, any over which need to be handled
    by the evaluation function.
    """
    # TODO: log these warnings/errors
    with np.errstate(all="raise"):
        try:
            if hasattr(ml_framework, "predict_proba"):
                pred = ml_framework.predict_proba(X)
                if task_type == classification_environments.MULTICLASS:
                    pred = _multiclass_predict_proba(pred)
                elif task_type == classification_environments.BINARY:
                    pred = _binary_predict_proba(pred)
                else:
                    pass
            elif hasattr(ml_framework, "predict"):
                pred = ml_framework.predict(X)
                if task_type == classification_environments.BINARY:
                    pred = pred
            else:
                raise NoPredictMethodError(
                    "ml_framework has no prediction function")
            return pred
        except FIT_PREDICT_ERRORS:
            return None
