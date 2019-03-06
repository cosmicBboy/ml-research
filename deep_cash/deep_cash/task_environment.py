"""A handler for generating tasks and datasets and evaluating ml frameworks."""

import logging
import numpy as np
import pynisher
import warnings

from functools import partial

from pynisher import TimeoutException, MemorylimitException, \
    CpuTimeoutException, SubprocessException, AnythingException
from sklearn.base import clone
from sklearn.preprocessing import label_binarize

from .data_environments import environments
from .data_environments.data_environment import NULL_DATA_ENV
from .data_types import FeatureType, TargetType, DataSourceType
from .errors import is_valid_fit_error, is_valid_predict_error, \
    FIT_WARNINGS, SCORE_WARNINGS, SCORE_ERRORS
from . import scorers, utils

logger = logging.getLogger(__name__)
FIT_GRACE_PERIOD = 30


PYNISHER_EXCEPTION = TimeoutException, MemorylimitException, \
    CpuTimeoutException, SubprocessException, AnythingException


class TaskEnvironment(object):
    """Generates datasets associated with supervised learning tasks."""

    def __init__(
            self,
            env_sources=["SKLEARN", "OPEN_ML", "KAGGLE"],
            test_env_sources=None,
            target_types=["BINARY", "MULTICLASS"],
            test_set_config=None,
            scorers=None,
            use_target_type_scorers=True,
            score_transformers=None,
            random_state=None,
            enforce_limits=True,
            per_framework_time_limit=10,
            per_framework_memory_limit=3072,
            dataset_names=None,
            test_dataset_names=None,
            error_reward=-0.1,
            n_samples=None,
            env_gen=environments.envs):
        """Initialize task environment.

        :param list[str] env_sources: list of data environment source names.
            These should correspond with the DataSourceType enum names.
        :param list[str]|None test_env_sources: list of data environment
            source names for test. These should correspond with the
            DataSourceType enum names.
        :param list[str] target_types: list of target types that the task
            environment will sample from. These should correspond with the
            TargetType enum names.
        :param dict[DataSourceType -> dict] test_set_config: a dictionary where
            keys DataSourceTypes and values are dictionaries of the form:
            {"test_size": float, "random_state": int}. This is used to set the
            task environment test set sizes.
        :param dict[TargetType: Scorer]|None scorers: where Scorer is a
            namedtuple composed of the following elements:
            - fn: an sklearn-compliant scoring function
            - reward_transformer: function with the signature (float) -> float
              that bounds range of the scoring function to have a range between
              0 and 1, inclusive.
            - comparator: a function with the signature (x, y) -> bool and
              returns True if x is a better score than y.
            If None, uses default scorers per target type.
        :param bool use_target_type_scorers: whether or not to use target-type-
            specific scorers specified in the `scorers` argument, or to use
            data-environment-specific scorers when available.
        :param int|None random_state: random seed determining the order of
            sampled data environments and the composition of the
            training/validation sets.
        :param bool enforce_limits: whether or not to enforce the per
            framework time limit and memory limit.
        :param int per_framework_time_limit: time limit in seconds of fit
            time limit (wallclock time). If exceeded during MLF fitting, the
            task environment returns an error reward.
        :param int per_framework_memory_limit: maximum amount of memory that
            can be allocated to a single ML framework fit. If exceeded during
            MLF fitting, the task environment returns an error reward.
        :param list[str] dataset_names: names of datasets to include in the
            task environment.
        :param list[str] test_dataset_names: names of datasets to exclude from
            the training task environment and should only be used in inference/
            evaluation of the controller.
        :param float error_reward: value to return when:
            - runtime raises an error during fitting.
            - fitting exceeds time and memory limit.
        :param int|None n_samples: number of samples to include in the data
            environment. If None, includes all samples. This should mainly be
            used for testing purposes to speed up fitting time.
        :param dict[DataSourceType, callable] env_source_map: data source type
            maps to a function with a signature

            (n: int, test_size: float, random_state: int|None, verbose: bool)
        """
        # TODO: need to add an attribute that normalizes the output of the
        # scorer function to support different tasks. Currently, the default
        # is ROC AUC scorer on just classification tasks. Will need to
        # extend this for regression task error metrics.

        if test_dataset_names is not None:
            dataset_intersection = set(dataset_names).intersection(
                set(test_dataset_names))
            if len(dataset_intersection) > 0:
                raise ValueError(
                    "dataset envs %s are also in test set. Training and test "
                    "data environments should be mutually exclusive" %
                    dataset_intersection)

        self._scorers = get_default_scorers() if scorers is None else scorers
        self._use_target_type_scorers = use_target_type_scorers
        self.random_state = random_state
        self.enforce_limits = enforce_limits
        self.per_framework_time_limit = per_framework_time_limit
        self.per_framework_memory_limit = per_framework_memory_limit

        test_set_config = {} if test_set_config is None else test_set_config
        test_set_config = {
            DataSourceType[k]: v for k, v in test_set_config.items()}

        test_env_sources = [] if test_env_sources is None else test_env_sources
        test_env_sources = [DataSourceType[e] for e in test_env_sources]

        # set train and test data environments
        get_envs = partial(
            env_gen,
            target_types=[TargetType[t] for t in target_types],
            test_set_config=test_set_config)
        self.data_distribution = get_envs(
            dataset_names, [DataSourceType[e] for e in env_sources])
        self.test_data_distribution = None
        if test_env_sources is not None:
            self.test_data_distribution = get_envs(
                test_dataset_names, test_env_sources)

        # TODO: the metafeature spec should be somehow persisted along with the
        # trained CASHController. Need to create a method in cash_reinforce
        # module that saves the data env, and test env training datasets.
        # This is required so that the data_env_name in the metafeature spec
        # is consistent at test time.
        self.metafeature_spec = utils.create_metafeature_spec(
            self.data_distribution, null_data_env=NULL_DATA_ENV)
        self.metafeature_dim = utils.get_metafeatures_dim(
            self.metafeature_spec)
        self.n_data_envs = len(self.data_distribution)
        self._error_reward = error_reward
        self._n_samples = n_samples
        self.current_data_env = None
        self._current_task = None
        self.create_metafeature_tensor = partial(
            utils._create_metafeature_tensor,
            metafeature_spec=self.metafeature_spec)

        # TODO: check if self.data_distribution is empty, if so raise
        # ValueError

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
        return [d.name for d in self.data_distribution]

    @property
    def error_reward(self):
        """Return reward in the situation of a fit/predict error."""
        return self._error_reward

    def sample_data_env(self):
        """Sample the data distribution."""
        self.set_data_env(self.data_distribution[
            np.random.randint(self.n_data_envs)])

    def set_data_env(self, data_env):
        """Set data environment from which to sample tasks."""
        self.current_data_env = data_env
        # TODO: probably want to standardize the scorer across tasks of
        # a certain target type, otherwise the reward function will very
        # complex
        if self._use_target_type_scorers or \
                self.current_data_env.scorer is None:
            self.scorer = self._scorers[self.current_data_env.target_type]
        else:
            self.scorer = self.current_data_env.scorer

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
                zip(self.current_data_env.feature_types,
                    self.current_data_env.feature_indices)
                if f == FeatureType.CATEGORICAL]
        }

    def sample_task_state(self, data_env_partition="train"):
        """Sample the current data_env for features and targets."""
        self._current_task = self.current_data_env.sample(self._n_samples)
        if data_env_partition == "train":
            name = self.current_data_env.name
        elif data_env_partition == "test":
            name = NULL_DATA_ENV
        return self.get_task_state(name, self._current_task.X_train)

    def get_test_task_state(self, data_env_partition="train"):
        if data_env_partition == "train":
            name = self.current_data_env.name
        elif data_env_partition == "test":
            # for test data envs, need to encode the data environment with the
            # null token, since it's a new category.
            name = NULL_DATA_ENV
        return self.get_task_state(name, self.current_data_env.X_test)

    def get_task_state(self, data_env_name, X):
        """Get state of task env, used as initial state of MLF proposal."""
        return self.create_metafeature_tensor(
            _metafeatures(data_env_name, X),
            # initial state is only a sequence of one item
            seq=[None])

    def evaluate(self, mlf):
        """Evaluate an ML framework by fitting and scoring on data.

        :params sklearn.pipeline.Pipeline mlf: an sklearn pipeline
        :returns: tuple where the first element is the reward the second
            element is the raw score.
        :rtype: 3-tuple of fitted mlf, reward, and validation score.
        """
        mlf = self._fit(mlf)
        reward, score = self._score_validation_set(mlf)
        return mlf, reward, score

    def _fit(self, mlf):
        """Fit proposed ML framework on currently sampled data environment.

        :params sklearn.pipeline.Pipeline mlf: an ML framework
            proposed by the CASH controller.
        :returns: sklearn.pipeline.Pipeline if MLF successfully fit, None
            if error was raised by calling `mlf.fit`.
        """
        mlf_str = utils._ml_framework_string(mlf)
        logger.info("FITTING MLF: %s" % mlf_str)

        try:
            result = self.ml_framework_fitter(
                clone(mlf),
                self._current_task.X_train,
                self._current_task.y_train)
            logger.info(
                "MLF FIT RESULT: %s" % ", ".join(["%s" % r for r in result]))
        except MemoryError as e:
            # catch memory error that may occur when dumping results from
            # the pynisher multiprocessing job.
            logger.info("ENCOUNTERED MEMORY ERROR: %s, MLF: %s, RESULT: %s" %
                        (e, mlf_str, ", ".join(["%s" % r for r in result])))

        try:
            mlf, fit_error = result
        except TypeError as e:
            logger.info("INVALID TYPE FIT ERROR: %s, MLF: %s, RESULT: %s" %
                        (e, mlf_str, result))
            return None

        if fit_error is None:
            return mlf
        elif self.enforce_limits and any(
                [fit_error is i for i in PYNISHER_EXCEPTION]):
            # if mlf fit routine incurred memory and runtime limit, task
            # environment should evaluate this as a negative reward.
            logger.info(
                "FIT LIMIT EXCEEDED: [%s] generated by mlf: %s, data env: %s" %
                (fit_error, mlf_str, self.current_data_env.name))
            return None
        elif is_valid_fit_error(fit_error):
            # if mlf fit routine raised valid fit error, task environment
            # should evaluate this as a negative reward.
            logger.info(
                "VALID FIT ERROR: [%s] generated by mlf %s, data env: %s" %
                (fit_error, mlf_str, self.current_data_env.name))
            return None
        else:
            # unrecognized fit errors should cause task environment to fail.
            # this is to improve the maintainability and predictability of
            # the pipeline so that all new errors that happen during MLF
            # evaluation are explicitly accounted for by the `errors` module.
            logger.exception(
                "INVALID FIT ERROR: [%s] generated by mlf %s, data env: %s" %
                (fit_error, mlf_str, self.current_data_env.name))
            return None

    def _score_validation_set(self, mlf):
        return self.score(
            mlf,
            self._current_task.X_validation,
            self._current_task.y_validation)

    def score(self, mlf, X, y):
        """Scores an MLF against some data."""
        logging.info("SCORING MLF: %s" % utils._ml_framework_string(mlf))
        none_return = None, None
        if mlf is None:
            return none_return

        y_hat = _ml_framework_predict(
            mlf, X, self.current_data_env.target_type, self.scorer.needs_proba)
        if y_hat is None:
            return none_return

        # need to reshape y and y_hat for multiclass cases
        if self.scorer.needs_proba and \
                self.current_data_env.target_type is TargetType.MULTICLASS:
            y = label_binarize(y, self.current_data_env.classes)
            # is y_hat is a one-dimensional array, assume that y_hat consists
            # of prediction labels that are reshaped to a binary array
            # representation.
            if len(y_hat.shape) == 1:
                y_hat = label_binarize(y_hat, self.current_data_env.classes)

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
                score = self.scorer.fn(y, y_hat)
            except SCORE_ERRORS:
                return none_return
            if self.scorer.reward_transformer is None:
                reward = score
            else:
                reward = self.scorer.reward_transformer(score)
        return reward, score


def get_default_scorers():
    # TODO: should scorers be a stochastic part of the environment? This
    # would mean that the controller would have to learn how maximize a
    # shifting reward function (model the reward function?)... might be
    # complicated.
    return {
        TargetType.BINARY: scorers.roc_auc(),
        TargetType.MULTICLASS: scorers.accuracy(),
        TargetType.REGRESSION: scorers.mean_squared_error(),
    }


def _metafeatures(data_env_name, X_train):
    """Create a metafeature vector.

    - data env name: categorical, e.g. "load_iris"
    - number of examples: continuous
    - number of features: continuous
    """
    return np.array([data_env_name, X_train.shape[0], X_train.shape[1]])


def _ml_framework_fitter(mlf, X, y):
    """Fits an ML framework to training data.

    This function handles warnings and errors

    :returns: a two-tuple where the first element is the proposed ML framework
        (sklearn.pipeline.Pipeline) and the second element is a subclass of
        BaseException if calling `mlf.fit` if it successfully fits a
        model and None if it fit successfully.
    """
    # TODO: handle MemoryError due to pynisher limits.
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
                return mlf.fit(X, y), None
            except Exception as error:
                return mlf, error


def _ml_framework_predict(mlf, X, target_type, needs_proba):
    """Generate predictions from a fit ml_framework.

    Handles errors at the predict layer, any over which need to be handled
    by the evaluation function.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with np.errstate(all="raise"):
            try:
                if needs_proba and hasattr(mlf, "predict_proba"):
                    pred = mlf.predict_proba(X)
                else:
                    pred = mlf.predict(X)
                return pred
            except Exception as error:
                mlf_str = utils._ml_framework_string(mlf)
                if is_valid_predict_error(error):
                    logger.info(
                        "VALID PREDICT ERROR: %s, no pipeline returned by "
                        "mlf framework %s" % (error, mlf_str))
                else:
                    logger.exception(
                        "INVALID PREDICT ERROR: ml framework pipeline: [%s], "
                        "error: \"%s\"" % (mlf_str, error))
                return None
