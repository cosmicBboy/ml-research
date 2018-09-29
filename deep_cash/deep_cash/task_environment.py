"""A handler for generating tasks and datasets and evaluating ml frameworks."""

import logging
import numpy as np
import operator
import pynisher
import warnings

from collections import namedtuple
from functools import partial

from sklearn.base import clone
from sklearn.metrics import f1_score, mean_squared_error

from .data_environments import environments
from .data_types import FeatureType, TargetType, DataSourceType
from .errors import NoPredictMethodError, is_valid_fit_error, \
    is_valid_predict_error, FIT_WARNINGS, SCORE_WARNINGS, SCORE_ERRORS
from . import utils

logger = logging.getLogger(__name__)
FIT_GRACE_PERIOD = 30

f1_score_weighted_average = partial(f1_score, average="weighted")

Scorer = namedtuple(
    "Scorer", [
        "fn",  # sklearn-compliant scoring function e.g. mean_squared_error
        "reward_transformer",  # function that bounds range of scoring function
        "comparator",  # function to determine if score a is better than b.
    ])


class TaskEnvironment(object):
    """Generates datasets associated with supervised learning tasks."""

    def __init__(
            self,
            env_sources=["SKLEARN", "OPEN_ML"],
            target_types=["BINARY", "MULTICLASS"],
            scorers=None,
            score_transformers=None,
            random_state=None,
            enforce_limits=True,
            per_framework_time_limit=10,
            per_framework_memory_limit=3072,
            dataset_names=None,
            error_reward=-0.1):
        """Initialize task environment."""
        # TODO: need to add an attribute that normalizes the output of the
        # scorer function to support different tasks. Currently, the default
        # is ROC AUC scorer on just classification tasks. Will need to
        # extend this for regression task error metrics.
        self.scorers = _get_scorers() if scorers is None else scorers
        self.random_state = random_state
        self.enforce_limits = enforce_limits
        self.per_framework_time_limit = per_framework_time_limit
        self.per_framework_memory_limit = per_framework_memory_limit
        self._dataset_names = dataset_names
        self._env_sources = [DataSourceType[e] for e in env_sources]
        self._target_types = [TargetType[t] for t in target_types]
        self.data_distribution = environments.envs(
            sources=self._env_sources, names=self._dataset_names,
            target_types=self._target_types)
        self.metafeature_spec = utils.create_metafeature_spec(
            self.data_distribution)
        self.metafeature_dim = utils.get_metafeatures_dim(
            self.metafeature_spec)
        self.n_data_envs = len(self.data_distribution)
        self._error_reward = error_reward
        self.create_metafeature_tensor = partial(
            utils._create_metafeature_tensor,
            metafeature_spec=self.metafeature_spec)

        # enforce resource contraints on framework fitter
        # TODO: support heuristic schedule for changing these contraints
        # over time.
        if self.enforce_limits:
            self.ml_framework_fitter = pynisher.enforce_limits(
                mem_in_mb=self.per_framework_memory_limit,
                wall_time_in_s=self.per_framework_time_limit,
                grace_period_in_s=FIT_GRACE_PERIOD)(_ml_framework_fitter)
        else:
            self.ml_framework_fitter = _ml_framework_fitter

        np.random.seed(self.random_state)

    @property
    def dataset_names(self):
        """Get dataset names in the task environment.

        :returns: a list of dataset names.
        :rtype: list[str]
        """
        return [d["dataset_name"] for d in self.data_distribution]

    @property
    def error_reward(self):
        """Return reward in the situation of a fit/predict error."""
        return self._error_reward

    def sample_data_env(self):
        """Sample the data distribution."""
        env_index = np.random.randint(self.n_data_envs)
        data_env = self.data_distribution[env_index]
        self.X = data_env["data"]
        self.y = data_env["target"]
        self.target_type = data_env["target_type"]
        self.data_env_name = data_env["dataset_name"]
        self.feature_types = data_env["feature_types"]
        self.feature_indices = data_env["feature_indices"]
        if data_env["target_preprocessor"] is not None:
            # NOTE: this feature isn't really executed currently since none of
            # the classification environments have a specified
            # target_preprocessor. This capability would be used in the case
            # of multi-label or multi-headed tasks.
            self.y = data_env["target_preprocessor"]().fit_transform(self.y)
        self.data_env_index = np.array(range(self.X.shape[0]))
        self.n_samples = len(self.data_env_index)

    def env_dep_hyperparameters(self):
        """Get data environment-dependent hyperparameters.

        Currently the only hyperparameter that dictionary would apply to is
        the OneHotEncoder component.

        This method is called in `cash_reinforce` to make sure that the
        pipeline one-hot-encodes categorical features. Should make a test for
        this.
        """
        return {
            "OneHotEncoder__categorical_features": [
                index for f, index in
                zip(self.feature_types, self.feature_indices)
                if f == FeatureType.CATEGORICAL]
        }

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
        return self.create_metafeature_tensor(
            _metafeatures(self.data_env_name, self.X_train, self.y_train),
            [None])

    def evaluate(self, mlf):
        """Evaluate an ML framework by fitting and scoring on data.

        :params sklearn.pipeline.Pipeline mlf: an sklearn pipeline
        :returns: tuple where the first element is the reward the second
            element is the raw score.
        :rtype: 2-tuple[float|None]
        """
        mlf = self._fit(mlf)
        return self._score(mlf)

    def _fit(self, mlf):
        """Fit proposed ML framework on currently sampled data environment.

        :params sklearn.pipeline.Pipeline mlf: an ML framework
            proposed by the CASH controller.
        :returns: sklearn.pipeline.Pipeline if MLF successfully fit, None
            if error was raised by calling `mlf.fit`.
        """
        mlf_str = utils._ml_framework_string(mlf)
        result = self.ml_framework_fitter(
            clone(mlf), self.X_train, self.y_train)
        try:
            mlf, fit_error = result
        except TypeError as e:
            logger.info("INVALID TYPE FIT ERROR: %s, MLF: %s, RESULT: %s" %
                        (e, mlf_str, result))
            return None

        if fit_error is None:
            return mlf
        elif self.enforce_limits and self.ml_framework_fitter.exit_status != 0:
            # if mlf fit routine incurred memory and runtime limit, task
            # environment should evaluate this as a negative reward.
            logger.info(
                "FIT LIMIT EXCEEDED: [%s] generated by mlf: %s" %
                (self.ml_framework_fitter.exit_status, mlf_str))
            return None
        elif is_valid_fit_error(fit_error):
            # if mlf fit routine raised valid fit error, task environment
            # should evaluate this as a negative reward.
            logger.info(
                "VALID FIT ERROR: [%s] generated by mlf %s" %
                (fit_error, mlf_str))
            return None
        else:
            # unrecognized fit errors should cause task environment to fail.
            # this is to improve the maintainability and predictability of
            # the pipeline so that all new errors that happen during MLF
            # evaluation are explicitly accounted for by the `errors` module.
            logger.exception(
                "INVALID FIT ERROR: [%s] generated by mlf %s" %
                (fit_error, mlf_str))
            return None

    def _score(self, mlf):
        if mlf is None:
            return None, None, None
        y_hat = _ml_framework_predict(mlf, self.X_test, self.target_type)
        scorer = self.scorers[self.target_type]
        if y_hat is None:
            return None, None, None
        with warnings.catch_warnings() as warning:
            # raise exception for warnings not explicitly in SCORE_WARNINGS
            warnings.filterwarnings("error")
            for warning_type, msg in SCORE_WARNINGS:
                # TODO: calling this every time an mlf is evaluated seems
                # inefficient... figure out a way of ignoring these warnings
                # once instead of with each call to _score
                warnings.filterwarnings(
                    "ignore", message=msg, category=warning_type)
            if warning:
                logger.info("SCORE WARNING: %s" % warning)
            try:
                score = scorer.fn(self.y_test, y_hat)
            except SCORE_ERRORS:
                return None, None, None
            if scorer.reward_transformer is None:
                reward = score
            else:
                reward = scorer.reward_transformer(score)
        return reward, score, scorer.comparator


def _get_scorers():
    return {
        TargetType.BINARY: Scorer(
            fn=f1_score_weighted_average,
            reward_transformer=None,
            comparator=operator.gt),
        TargetType.MULTICLASS: Scorer(
            fn=f1_score_weighted_average,
            reward_transformer=None,
            comparator=operator.gt),
        TargetType.REGRESSION: Scorer(
            fn=mean_squared_error,
            reward_transformer=exponentiated_log,
            comparator=operator.lt),
    }


def exponentiated_log(x, gamma=0.01):
    """Bounds functions with a range of >= 0 to range of [0, 1].

    :param float x: value to transform
    :param float gamma: decay coefficient. Larger gamma means function decays
        more quickly as x gets larger.
    :returns: transformed value.
    :rtype: float
    """
    if x < 0:
        raise ValueError("value %s not a valid input. Must be >= 0")
    if x == 0:
        # since the below function is undefined at x=0, return 1 if x=0.
        return 1.0
    return 1 / (1 + np.power(np.e, np.log(gamma * x)))


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

    :returns: a two-tuple where the first element is the proposed ML framework
        (sklearn.pipeline.Pipeline) and the second element is a subclass of
        BaseException if calling `ml_framework.fit` if it successfully fits a
        model and None if it fit successfully.
    """
    # raise numpy overflow errors
    with warnings.catch_warnings():
        # TODO: calling this every time an mlf is evaluated seems
        # inefficient... figure out a way of ignoring these warnings
        # once instead of with each call to _score
        for warning_type, msg in FIT_WARNINGS:
            warnings.filterwarnings(
                "ignore", message=msg, category=warning_type)
        with np.errstate(all="raise"):
            try:
                return ml_framework.fit(X, y), None
            except Exception as error:
                return ml_framework, error


def _binary_predict_proba(pred_proba):
    # TODO: see note in _score method above. Keeping this function for now even
    # though it seems redundant with _multiclass_predict_proba
    return pred_proba.argmax(axis=1)


def _multiclass_predict_proba(pred_proba):
    # TODO: see note in _score method above. Keeping this function for now even
    # though it seems reduntant with _binary_predict_proba
    return pred_proba.argmax(axis=1)


def _ml_framework_predict(ml_framework, X, target_type):
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
                    if target_type == TargetType.MULTICLASS:
                        pred = _multiclass_predict_proba(pred)
                    elif target_type == TargetType.BINARY:
                        pred = _binary_predict_proba(pred)
                    else:
                        pass
                elif hasattr(ml_framework, "predict"):
                    pred = ml_framework.predict(X)
                    if target_type == TargetType.BINARY:
                        pred = pred
                else:
                    raise NoPredictMethodError(
                        "ml_framework has no prediction function")
                return pred
            except Exception as error:
                mlf_str = utils._ml_framework_string(ml_framework)
                if is_valid_predict_error(error):
                    logger.info(
                        "VALID PREDICT ERROR: %s, no pipeline returned by "
                        "mlf framework %s" % (error, mlf_str))
                else:
                    logger.exception(
                        "INVALID PREDICT ERROR: ml framework pipeline: [%s], "
                        "error: \"%s\"" % (mlf_str, error))
                return None
