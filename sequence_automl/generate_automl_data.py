"""Module for generating auto-ml data.

This module should provide an interface to generate the following training data

create a set of utility functions that creates the following tuples:
(is_executable, creates_estimator, creates_pipeline, code_string)

TODO:
- create a list of strings that evaluate to legitimate calls to the sklearn
  API, but don't evaluate to an estimator, e.g. `sklearn.linear_model`.
  - label these instances as `is_executable = 1`, `creates_estimator = 0`,
    and `creates_pipeline = 0`.
- create a list of strings that evaluate to legitimate calls
"""


import inspect
import sklearn
import warnings

from collections import namedtuple
from sklearn.utils.testing import all_estimators


AlgorithmEnv = namedtuple(
    "AlgorithmEnv", [
        "classifiers", "regressors", "transformers",
        "classifier_hyperparams", "regressor_hyperparams",
        "transformer_hyperparams"])
HyperparameterSpec = namedtuple(
    "HyperparameterSpec", ["hyperparameters", "defaults"])


def create_algorithm_env():
    """Create Algorithm Environment, used to sample algorithms, hyperparameters

    TODO: This should probably implemented as a class so that methods can be
    better organized around this concept.

    Returns
    -------
    namedtuple AlgorithmEnv containing dictionaries specifying the structure
        of the environment.
    """
    classifiers = get_estimators("classifier")
    regressors = get_estimators("regressor")
    transformers = get_estimators("transformer")
    classifier_hyperparams = {
        c[0]: get_hyperparams(c[1]) for c in classifiers.items()}
    regressor_hyperparams = {
        r[0]: get_hyperparams(r[1]) for r in regressors.items()}
    transformer_hyperparams = {
        t[0]: get_hyperparams(t[1]) for t in transformers.items()}
    return AlgorithmEnv(
        classifiers, regressors, transformers,
        classifier_hyperparams, regressor_hyperparams, transformer_hyperparams)


def get_estimators(estimator_type):
    """Get estimators of a particular type.

    Parameters
    ----------
    estimator_type: string {'classifier', 'regressor', 'transformer'}

    Returns
    -------
    dict[str -> object] where keys are the name of the estimator and values
        are the corresponding class.
    """
    with warnings.catch_warnings():
        # catch deprecation warning for modules to be removed in 0.20 in favor
        # of the model_selection module: cross_validation, grid_search, and
        # learning_curve
        warnings.simplefilter("ignore")
        return dict(all_estimators(type_filter=estimator_type))


def get_hyperparams(EstimatorTransformer):
    """Get hyperparameters and their defaults from an estimator class.

    Parameters
    ----------
    EstimatorTransformer: sklearn estimator or transformer class

    Returns
    -------
    HyperparameterSpec namedtuple with attributes:
        - hyperparameters list[str]
        - default tuple[str|int|float|None|...]
    """
    argspec = inspect.getargspec(EstimatorTransformer)
    hyperparams = [arg for arg in argspec.args if arg != "self"]
    if len(hyperparams) > 0 and argspec.defaults is not None:
        assert len(hyperparams) == len(argspec.defaults)
    return HyperparameterSpec(hyperparams, argspec.defaults)


def get_feature_preprocessors():
    pass


def get_data_preprocessors():
    pass


def eval_is_executable(estimator_code):
    pass


def eval_is_estimator(estimator_code):
    pass


if __name__ == "__main__":
    algorithm_env = create_algorithm_env()
    print("Algorithm Environment:")
    print("classifiers: %s" % len(algorithm_env.classifiers))
    print("regressors: %s" % len(algorithm_env.regressors))
    print("transformers: %s" % len(algorithm_env.transformers))
