"""A handler for generating tasks and datasets and evaluating ml frameworks."""

import logging
import numpy as np
import pynisher
import warnings

from functools import partial

from sklearn.base import clone
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning

from .data_environments import classification_environments
from .errors import NoPredictMethodError, ExceededResourceLimitError, \
    is_valid_fit_error, is_valid_predict_error, FIT_ERROR_TYPES, \
    FIT_WARNINGS, PREDICT_ERROR_TYPES, SCORE_WARNINGS
from . import utils

LOGGER = utils.init_logging(__file__)

FIT_GRACE_PERIOD = 30


class TaskEnvironment(object):
    """Generates datasets associated with supervised learning tasks."""

    def __init__(
            self, scorer=f1_score, scorer_kwargs=None, random_state=None,
            per_framework_time_limit=10, per_framework_memory_limit=3072,
            error_reward=-0.1, reward_scale=10, reward_transformer=None):
        """Initialize task environment."""
        # TODO: need to add an attribute that normalizes the output of the
        # scorer function to support different tasks. Currently, the default
        # is ROC AUC scorer on just classification tasks. Will need to
        # extend this for regression task error metrics.
        self.scorer = partial(scorer, **scorer_kwargs)
        self.random_state = random_state
        self.per_framework_time_limit = per_framework_time_limit
        self.per_framework_memory_limit = per_framework_memory_limit
        self.data_distribution = classification_environments.envs()
        self.n_data_envs = len(self.data_distribution)
        self._error_reward = error_reward
        self._reward_scale = reward_scale
        self._reward_transformer = reward_transformer

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
        return self._reward_transformer(self._error_reward)

    @property
    def correct_hyperparameter_reward(self):
        """Return reward when proposed hyperparameter is correct.

        TODO: Deprecate this function when removing old
        rnn_ml_framework_controller.py scripts
        """
        return 1 * self._reward_scale

    def sample_data_env(self):
        """Sample the data distribution."""
        env_index = np.random.randint(self.n_data_envs)
        data_env_tuple = self.data_distribution[env_index]
        env_name, env_fn, task_type, target_preprocessor = data_env_tuple
        data_env = env_fn()
        self.X = data_env["data"]
        self.y = data_env["target"]
        self.task_type = task_type
        self.data_env_name = env_name
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
        return _metafeatures(
            self.data_env_name,
            self.X_train,
            self.y_train)

    def evaluate(self, ml_framework):
        """Evaluate an ML framework by fitting and scoring on data."""
        return self._fit_score(ml_framework)

    def get_reward(self, r):
        if self._reward_transformer:
            return self._reward_transformer(r)
        return r

    def _fit(self, ml_framework):
        # TODO: log these warnings/errors
        mlf = clone(ml_framework)
        ml_framework = self.ml_framework_fitter(
            mlf, self.X_train, self.y_train)
        if self.ml_framework_fitter.exit_status != 0:
            msg = "ml framework %s exceeded with status: %s" % (
                ml_framework, self.ml_framework_fitter.exit_status)
            print(msg)
            LOGGER.exception(msg)
            return None
        return ml_framework

    def _score(self, ml_framework):
        # TODO: the scorer should dictate the form that the prediction takes.
        # for example, the f1_score can only take categorical predictions, not
        # predicted probabilities. For now, all predictions are treated as
        # categorical
        pred = _ml_framework_predict(
            ml_framework, self.X_test, self.task_type)
        if pred is None:
            return None
        with warnings.catch_warnings() as w:
            for warning_type, msg in SCORE_WARNINGS:
                warnings.filterwarnings(
                    "ignore", message=msg, category=warning_type)
            if w:
                print(w)
            return self.get_reward(self.scorer(self.y_test, pred))

    def _fit_score(self, ml_framework):
        mlf = self._fit(ml_framework)
        if mlf is None:
            return None
        return self._score(mlf)


def _metafeatures(data_env_name, X_train, y_train):
    """Create a metafeature vector.

    - data env name: categorical, e.g. "load_iris"
    - number of examples: continuous
    - number of features: continuous
    """
    return np.array([data_env_name, X_train.shape[0], X_train.shape[1]])


def _ml_framework_fitter(ml_framework, X, y):
    """Fits an ML framework to training data.

    This function handles warnings and errors

    :returns: a valid ml_framework if it successfully fits a model, None if
        calling the `fit` method leads to
    """
    # raise numpy overflow errors
    # TODO: log these warnings/errors
    with warnings.catch_warnings():
        for warning_type, msg in FIT_WARNINGS:
            warnings.filterwarnings(
                "ignore", message=msg, category=warning_type)
        with np.errstate(all="raise"):
            try:
                ml_framework.fit(X, y)
                return ml_framework
            except FIT_ERROR_TYPES as error:
                if is_valid_fit_error(error):
                    return None
                print("error fit predict")
                print(type(error), error)
                LOGGER.exception(
                    "FIT ERROR: ml framework pipeline: [%s], error: "
                    "<<<\"%s\">>>" % (
                        utils._ml_framework_string(ml_framework), error))
                raise
            except Exception as error:
                raise


def _binary_predict_proba(pred_proba):
    # TODO: see note in _score method above. Keeping this function for now even
    # though it seems reduntant with _multiclass_predict_proba
    return pred_proba.argmax(axis=1)


def _multiclass_predict_proba(pred_proba):
    # TODO: see note in _score method above. Keeping this function for now even
    # though it seems reduntant with _multiclass_predict_proba
    return pred_proba.argmax(axis=1)


def _ml_framework_predict(ml_framework, X, task_type):
    """Generate predictions from a fit ml_framework.

    Handles errors at the predict layer, any over which need to be handled
    by the evaluation function.
    """
    # TODO: see note in _score method above.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
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
            except PREDICT_ERROR_TYPES as error:
                if is_valid_predict_error(error):
                    return None
                LOGGER.exception(
                    "PREDICT ERROR: ml framework pipeline: [%s], error: \"%s\""
                    % (utils._ml_framework_string(ml_framework), error))
                raise
            except Exception:
                raise
