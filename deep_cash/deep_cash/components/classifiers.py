"""Classifier components."""

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from .algorithm import AlgorithmComponent
from .hyperparameter import (
    CategoricalHyperparameter, UniformIntHyperparameter,
    UniformFloatHyperparameter)
from . import constants


def logistic_regression():
    """Create a logistic regression algorithm component.

    Leave out the following hyperparameters:
    - intercept_scaling
    - random_state
    - warm_start
    - max_iter
    - tol
    """
    return AlgorithmComponent(
        "LogisticRegression", LogisticRegression, constants.CLASSIFIER, [
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
                "multi_class", ["ovr", "multinomial"], default="ovr"),
            UniformIntHyperparameter("C", 1, 300, default=1.0, log=True, n=5),
        ])


def gaussian_naive_bayes():
    """Create a naive bayes algorithm component."""
    return AlgorithmComponent(
        "GaussianNaiveBayes", GaussianNB, constants.CLASSIFIER,)


def decision_tree():
    """Create a decision tree algorithm component.

    Note here that if the hyperparameter can be either str, int, or float,
    default to float. Future versions will support MultiTypeHyperparameters.

    Leave out the following hyperparameters:
    - min_impurity_split (deprecated)
    - random_state
    - presort (mostly for speed up)
    """
    return AlgorithmComponent(
        "DecisionTreeClassifier", DecisionTreeClassifier,
        constants.CLASSIFIER, [
            CategoricalHyperparameter(
                "criterion", ["gini", "entropy"], default="gini"),
            CategoricalHyperparameter(
                "splitter", ["best", "random"], default="best"),
            CategoricalHyperparameter(
                "class_weight", ["balanced", None], default=None),
            UniformIntHyperparameter(
                "max_depth", 1, 1000, default=None, log=True),
            UniformFloatHyperparameter(
                "min_samples_split", 0.0, 1.0, default=0.05),
            UniformFloatHyperparameter(
                "min_samples_leaf", 0.0, 1.0, default=1.0),
            UniformFloatHyperparameter(
                "min_weight_fraction_leaf", 0.0, 1.0, default=0.0),
            UniformFloatHyperparameter(
                "max_features", 0.0, 1.0, default=0.33),
            UniformIntHyperparameter(
                "max_leaf_nodes", 1, 1000, default=None, log=True),
            UniformFloatHyperparameter(
                "min_impurity_decrease", 0.0, 1.0, default=0.0, n=10),
        ])
