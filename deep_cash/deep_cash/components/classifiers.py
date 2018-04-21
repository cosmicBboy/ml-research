"""Classifier components."""

from sklearn.linear_model import LogisticRegression

from .algorithm import AlgorithmComponent
from .hyperparameter import (
    Hyperparameter, CategoricalHyperparameter, UniformIntHyperparameter,
    UniformFloatHyperparameter)


def logistic_regression():
    """Creates a logistic regression algorithm component.

    Leave out the following hyperparameters:
    - intercept_scaling
    - random_state
    - warm_start
    """
    return AlgorithmComponent(
        "LogisticRegression", LogisticRegression, [
            CategoricalHyperparameter("penalty", ["l1", "l2"], default="l2"),
            CategoricalHyperparameter("dual", [True, False], default=False),
            CategoricalHyperparameter(
                "fit_intercept", [True, False], default=True),
            CategoricalHyperparameter(
                "class_weight", ["balanced", None], default=None),
            CategoricalHyperparameter(
                "solver", ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
                default="liblinear"),
            CategoricalHyperparameter(
                "multiclass", ["ovr", "multinomial"], default="ovr"),
            UniformFloatHyperparameter(
                "tol", 1e-6, 1e-3, default=1e-4, log=True, n=4),
            UniformIntHyperparameter("C", 1, 300, default=1.0, log=True, n=5),
            UniformIntHyperparameter("max_iter", 100, 300, default=100),
        ])
