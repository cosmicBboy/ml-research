"""Module for generating auto-ml data.

TODO: move this to its own project at top level `ml-research` root

This module should provide an interface to generate the following training data

create a set of utility functions that creates the following tuples:
(is_executable, creates_estimator, creates_pipeline, code_string)
"""


import inspect
import numpy as np
import random
import re
import sklearn
import string
import warnings

from collections import namedtuple
from sklearn.utils.testing import all_estimators


CHARACTERS = string.ascii_letters + string.digits + "._()"
# These estimators have been deprecated
EXCLUDE_ESTIMATORS = [
    "RandomizedLogisticRegression",
    "RandomizedLasso",
    "RandomizedPCA",
    "DictionaryLearning",
    "MiniBatchDictionaryLearning",
    "GaussianProcess",
]
EXECUTABLE_MAP = {
    True: "executable",
    False: "not_executable",
}
CREATES_ESTIMATOR = {
    True: "creates_estimator",
    False: "not_creates_estimator",
}


HyperparameterSpec = namedtuple(
    "HyperparameterSpec", ["hyperparameters", "defaults"])


class AlgorithmEnv(object):

    def __init__(self, classifiers, regressors, transformers):
        """Initialize Algorithm and Hyperparameter Space (AHS).

        The AlgorithmEnv is the environment from which we sample algorithms,
        which are defined in this context as sklearn Transformer and Estimator
        objects.

        :param list[AlgorithmSpec] classifiers: list of classifiers to search
        :param list[AlgorithmSpec] regressors: list of regressors to search
        :param list[AlgorithmSpec] transformers: list of transformers to search
        """
        self.classifiers = classifiers
        self.regressors = regressors
        self.transformers = transformers

    def sample_algorithm_code(self, type_filter=None):
        """Samples an algorithm."""
        sample_code = self._sample_algorithm_code(type_filter)
        return self._input_data(sample_code)

    def algorithm_obj_to_instance(self, sample):
        """Convert algorithm code so that code evaluates to instance."""
        _, sample_code = sample
        return self._input_data("%s()" % sample_code)

    def mutate_sample(self, sample, mutate_all=False):
        """Mutates a sample.

        :param tuple sample: (metafeatures <list>, sample_code <str>)
        :returns: mutated sample
        """
        _, sample_code = sample
        return self._input_data(
            self._mutate_sample_code(sample_code, mutate_all))

    def _sample_algorithm_code(self, type_filter):
        """Samples an algorithm from the algorithm environment.

        :param str type_filter: {"classifiers", "regressors", "transformers"}
        :returns BaseEstimator: an sklearn algorithm (estimator/transformer)
        """
        algorithm_sample_space = None
        if type_filter is None:
            algorithm_sample_space = self.classifiers + self.regressors + \
                self.transformers
        elif type_filter == "classifiers":
            algorithm_sample_space = self.classifiers
        elif type_filter == "regressors":
            algorithm_sample_space = self.regressors
        elif type_filter == "transformers":
            algorithm_sample_space = self.transformers
        else:
            raise ValueError(
                "unrecognized type_filter argument: %s" % type_filter)
        return random.choice(algorithm_sample_space).estimator_code

    def _mutate_sample_code(self, sample_code, mutate_all):
        """Randomly mutate a random number of characters in the code."""
        code_length = len(sample_code)
        mutate_n = len(code_length) if mutate_all else \
            random.randint(0, code_length)
        mutate_indices = np.random.choice(
            range(code_length), mutate_n, replace=False)
        sample_code_split = [s for s in sample_code]
        for i in mutate_indices:
            char = CHARACTERS[random.randint(0, len(CHARACTERS) - 1)]
            sample_code_split[i] = char
        return "".join(sample_code_split)

    def _input_data(self, sample_code):
        """Creates a data sample in the form (metafeatures, sample_code)."""
        is_executable = code_is_executable(sample_code)
        evals_to_algorithm_instance = False
        if is_executable:
            evals_to_algorithm_instance = code_evals_to_algorithm_instance(
                sample_code)
        metafeatures = [
            EXECUTABLE_MAP[is_executable],
            CREATES_ESTIMATOR[evals_to_algorithm_instance]]
        return (metafeatures, sample_code)


class AlgorithmSpec(object):

    def __init__(self, estimator_name, estimator_class):
        self.estimator_name = estimator_name
        self.estimator_class = estimator_class

    @property
    def hyperparameters(self):
        return self._get_hyperparams(self.estimator_class)

    @property
    def estimator_code(self):
        return self._get_estimator_string(self.estimator_class)

    def _get_hyperparams(self, estimator_class):
        """Get hyperparameters and their defaults from an estimator class.

        :param BaseEstimator estimator_class: sklearn estimator or transformer
            class.
        :returns HyperparameterSpec: a specification of the
            hyperparameter space, with attributes:
            - hyperparameters: list[str]
            - default: tuple[str|int|float|None|...]
        """
        argspec = inspect.getfullargspec(estimator_class)
        hyperparams = [arg for arg in argspec.args if arg != "self"]
        if len(hyperparams) > 0 and argspec.defaults is not None:
            assert len(hyperparams) == len(argspec.defaults)
        return HyperparameterSpec(hyperparams, argspec.defaults)

    def _get_estimator_string(self, estimator_class):
        """Extract string representation from Estimator class."""
        return re.search(
            "<class '(.+)'>", str(estimator_class)).groups()[0]

    def __repr__(self):
        return "AlgorithmSpec__%s" % self.estimator_name



def create_algorithm_env():
    """Create Algorithm Environment, used to sample algorithms, hyperparameters

    TODO: This should probably implemented as a class so that methods can be
    better organized around this concept.

    :returns AlgorithmEnv: an object containing dictionaries specifying the
        search space of the environment.
    """
    return AlgorithmEnv(
        get_algorithms("classifier"),
        get_algorithms("regressor"),
        get_algorithms("transformer"))


def get_algorithms(estimator_type):
    """Get estimators of a particular type.

    This function relies on the ``all_estimators`` function in
    sklearn.utils.testing.

    :param str estimator_type: {'classifier', 'regressor', 'transformer'}
        specifies the type of estimator to get.
    :returns dict[str, object]: where keys are the name of the estimator and
        values are the corresponding class.
    """
    with warnings.catch_warnings():
        # catch deprecation warning for modules to be removed in 0.20 in favor
        # of the model_selection module: cross_validation, grid_search, and
        # learning_curve
        warnings.simplefilter("ignore")
        return [
            AlgorithmSpec(estimator_name, estimator_class) for
            estimator_name, estimator_class in
            all_estimators(type_filter=estimator_type)
            if estimator_name not in EXCLUDE_ESTIMATORS]


def code_is_executable(estimator_code):
    """Determine if code string is executable.

    :param str estimator_code: string to check.
    :returns bool: True if executable
    """
    try:
        eval(estimator_code)
        return True
    except:
        return False


def code_evals_to_algorithm_instance(estimator_code):
    """Determing if code evaluates to a BaseEstimator instance."""
    parent_classes = inspect.getmro(type(eval(estimator_code)))
    return sklearn.base.BaseEstimator in parent_classes


if __name__ == "__main__":
    algorithm_env = create_algorithm_env()
    print("Algorithm Environment:")
    print("classifiers: %s" % len(algorithm_env.classifiers))
    print("regressors: %s" % len(algorithm_env.regressors))
    print("transformers: %s" % len(algorithm_env.transformers))

    print("\nGenerating sample data from environment:")
    for _ in range(5):
        sample_data = algorithm_env.sample_algorithm_code()
        print(sample_data)
        print(algorithm_env.algorithm_obj_to_instance(sample_data))
        print(algorithm_env.mutate_sample(sample_data))
        print(algorithm_env.mutate_sample(sample_data, mutate_all=False))
        print("")
