"""Classifier components."""

from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, \
    GradientBoostingClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier

from .algorithm_component import AlgorithmComponent, EXCLUDE_ALL
from .hyperparameter import (
    CategoricalHyperparameter, UniformIntHyperparameter,
    UniformFloatHyperparameter, BaseEstimatorHyperparameter)
from ..data_types import AlgorithmType

# TODO: add sgd classifier, xgboost classifier


# ===========
# Naive Bayes
# ===========

def gaussian_naive_bayes():
    """Create a naive bayes algorithm component."""
    return AlgorithmComponent(
        name="GaussianNaiveBayes",
        component_class=GaussianNB,
        component_type=AlgorithmType.CLASSIFIER)


def multinomial_naive_bayes():
    """Create a naive bayes algorithm component."""
    return AlgorithmComponent(
        name="MultinomialNB",
        component_class=MultinomialNB,
        component_type=AlgorithmType.CLASSIFIER,
        hyperparameters=[
            UniformFloatHyperparameter(
                "alpha", 1e-2, 100, default=1, log=True),
            CategoricalHyperparameter(
                "fit_prior", [True, False], default=True)
        ])


# ====================
# Gaussian Classifiers
# ====================


def rbf_gaussian_process_classifier():
    return AlgorithmComponent(
        name="GaussianProcessClassifier",
        component_class=GaussianProcessClassifier,
        component_type=AlgorithmType.CLASSIFIER,
        hyperparameters=[
            UniformFloatHyperparameter(
                "max_iter_predict", 100, 500, default=100)
        ],
        constant_hyperparameters={
            "optimizer": "fmin_l_bfgs_b",
            "n_restarts_optimizer": 10,
        })


# ========================
# Linear Model Classifiers
# ========================

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
        name="LogisticRegression",
        component_class=LogisticRegression,
        component_type=AlgorithmType.CLASSIFIER,
        hyperparameters=[
            CategoricalHyperparameter("penalty", ["l1", "l2"], default="l2"),
            CategoricalHyperparameter(
                "dual", [True, False], default=False),
            CategoricalHyperparameter(
                "fit_intercept", [True, False], default=True),
            CategoricalHyperparameter(
                "class_weight", ["balanced", None], default=None),
            CategoricalHyperparameter(
                "solver", ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
                default="liblinear"),
            CategoricalHyperparameter(
                "multi_class", ["ovr", "multinomial"], default="ovr"),
            UniformFloatHyperparameter(
                "C", 0.001, 300.0, default=1.0, log=True, n=10),
        ],
        exclusion_conditions={
            "penalty": {
                "l1": {"dual": [True],
                       "solver": ["newton-cg", "lbfgs", "sag"]}},
            "dual": {True: {"solver": ["newton-cg", "lbfgs", "sag", "saga"]}},
            "solver": {"liblinear": {"multi_class": ["multinomial"]}},
        })


# ==========================
# Non-parametric Classifiers
# ==========================

def k_nearest_neighbors():
    """Create K-nearest neighbors classifier component.

    Going to rely on the `algorithm="auto"` and the other default settings,
    following the auto-sklearn implementation:
    https://github.com/automl/auto-sklearn/blob/master/autosklearn/pipeline/components/classification/k_nearest_neighbors.py  # noqa

    The hyperparameters that were left out were:
    - algorithm {'auto', 'ball_tree', 'kd_tree', 'brute'}
    - leaf_size (int)
    - p (int)
    - metric: distance metric to use for tree (default 'minkowski')
    - metric_params: additional keyword arguments for the metric function.
    """
    return AlgorithmComponent(
        name="KNearestNeighorsClassifier",
        component_class=KNeighborsClassifier,
        component_type=AlgorithmType.CLASSIFIER,
        hyperparameters=[
            UniformIntHyperparameter(
                "n_neighbors", 1, 100, log=True, default=1),
            CategoricalHyperparameter(
                "weights", ["uniform", "distance"], default="uniform"),
        ])


# ==========================
# Support-vector Classifiers
# ==========================

def support_vector_classifier_linear():
    """Create linear support vector classifier component.

    hyperparameter settings were taken from:
    https://github.com/automl/auto-sklearn/blob/master/autosklearn/pipeline/components/classification/liblinear_svc.py  # noqa

    See scikit-learn docs for details on constant hyperparameters:
    http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html

    TODO: implement the cache size logic as described in the auto-sklearn
    project https://github.com/automl/auto-sklearn/blob/master/autosklearn/pipeline/components/classification/libsvm_svc.py  # noqa
    """
    return AlgorithmComponent(
        name="LinearSVC",
        component_class=LinearSVC,
        component_type=AlgorithmType.CLASSIFIER,
        hyperparameters=[
            CategoricalHyperparameter("penalty", ["l1", "l2"], default="l2"),
            CategoricalHyperparameter(
                "loss", ["hinge", "squared_hinge"], default="squared_hinge"),
            UniformIntHyperparameter(
                "max_iter", 1000, 3000, default=1000),
            UniformFloatHyperparameter(
                "tol", 1e-5, 1e-1, default=1e-4, log=True),
            UniformFloatHyperparameter(
                "C", 0.03215, 32768, log=True, default=1.0)
        ],
        constant_hyperparameters={
            "multi_class": "ovr",
            "dual": False
        },
        exclusion_conditions={
            "penalty": {
                "l1": {"loss": ["hinge"]},
                "l2": {"loss": ["hinge"]},
            },
        })


def support_vector_classifier_nonlinear():
    """Create nonlinear support vector classifier (rbf, poly, sigmoid)"""
    return AlgorithmComponent(
        name="NonlinearSVC",
        component_class=SVC,
        component_type=AlgorithmType.CLASSIFIER,
        hyperparameters=[
            UniformFloatHyperparameter(
                "C", 0.03125, 32768, default=1.0, log=True),
            CategoricalHyperparameter(
                "kernel", ["rbf", "poly", "sigmoid"], default="rbf"),
            UniformIntHyperparameter("degree", 2, 5, default=3),
            UniformFloatHyperparameter(
                "gamma", 3.0517578125e-05, 8, default=0.1, log=True),
            UniformFloatHyperparameter("coef0", -1, -1, default=0),
            CategoricalHyperparameter(
                "shrinking", [True, False], default=True),
            UniformFloatHyperparameter(
                "tol", 1e-5, 1e-1, default=1e-3, log=True),
        ],
        constant_hyperparameters={
            "max_iter": -1,
            "cache_size": 200,
            "decision_function_shape": "ovr",
        },
        exclusion_conditions={
            "kernel": {
                "rbf": {
                    "degree": EXCLUDE_ALL,
                    "coef0": EXCLUDE_ALL,
                },
                "sigmoid": {"degree": EXCLUDE_ALL}
            },
        })


# ======================
# Tree-based Classifiers
# ======================

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
        name="DecisionTreeClassifier",
        component_class=DecisionTreeClassifier,
        component_type=AlgorithmType.CLASSIFIER,
        hyperparameters=[
            CategoricalHyperparameter(
                "criterion", ["gini", "entropy"], default="gini"),
            CategoricalHyperparameter(
                "splitter", ["best", "random"], default="best"),
            CategoricalHyperparameter(
                "class_weight", ["balanced", None], default=None),
            UniformIntHyperparameter(
                "max_depth", 1, 1000, default=None, log=True),
            UniformFloatHyperparameter(
                "min_samples_split", 0.01, 1.0, default=0.05),
            UniformFloatHyperparameter(
                "min_samples_leaf", 0.01, 0.5, default=0.1),
            UniformFloatHyperparameter(
                "min_weight_fraction_leaf", 0.01, 0.5, default=0.0),
            UniformFloatHyperparameter(
                "max_features", 0.0, 1.0, default=0.33),
            UniformIntHyperparameter(
                "max_leaf_nodes", 2, 1000, default=None, log=True),
            UniformFloatHyperparameter(
                "min_impurity_decrease", 0.0, 1.0, default=0.0, n=10),
        ])


# ====================
# Ensemble Classifiers
# ====================

def adaboost():
    """Create adaboost classifier component."""
    return AlgorithmComponent(
        name="AdaBoostClassifier",
        component_class=AdaBoostClassifier,
        component_type=AlgorithmType.CLASSIFIER,
        hyperparameters=[
            BaseEstimatorHyperparameter(
                hname="base_estimator",
                base_estimator=DecisionTreeClassifier,
                hyperparameters=[
                    UniformIntHyperparameter("max_depth", 1, 10, default=1)
                ],
                default=DecisionTreeClassifier(max_depth=1)
            ),
            UniformIntHyperparameter(
                "n_estimators", 10, 200, default=10, n=10),
            UniformFloatHyperparameter(
                "learning_rate", 0.1, 1., default=1., log=True, n=10),
            CategoricalHyperparameter(
                "algorithm", ["SAMME", "SAMME.R"], default="SAMME.R")
        ])


def extra_trees():
    """Create extra trees classifier component.

    Just follows:
    https://github.com/automl/auto-sklearn/blob/master/autosklearn/pipeline/components/classification/extra_trees.py  # noqa
    """
    return ExtraTreesClassifier(
        name="ExtraTreesClassifier",
        component_class=ExtraTreesClassifier,
        component_type=AlgorithmType.CLASSIFIER,
        hyperparameters=[
            CategoricalHyperparameter(
                "criterion", ["gini", "entropy"], default="gini"),
            UniformFloatHyperparameter(
                "max_features", 0., 1., default=0.5),
            UniformIntHyperparameter("min_samples_split", 2, 20, default=2),
            UniformIntHyperparameter("min_samples_leaf", 1, 20, default=1),
            CategoricalHyperparameter(
                "bootstrap", [True, False], default=False),
            CategoricalHyperparameter(
                "class_weight", ["balanced", "balanced_subsample", None],
                default=None),
        ],
        constant_hyperparameters={
            "n_estimators": 100,
            "max_depth": None,
            "min_weight_fraction_leaf": 0.,
            "max_leaf_nodes": None,
            "min_impurity_decrease": 0.,
        })


def gradient_boosting():
    """Create gradient boosting classifier component."""
    return AlgorithmComponent(
        name="GradientBoostingClassifier",
        component_class=GradientBoostingClassifier,
        component_type=AlgorithmType.CLASSIFIER,
        hyperparameters=[
            UniformFloatHyperparameter(
                "learning_rate", 0.01, 1, default=0.1, log=True),
            UniformIntHyperparameter("n_estimators", 50, 500, default=100),
            UniformIntHyperparameter("max_depth", 1, 10, default=3),
            CategoricalHyperparameter(
                "criterion", ["friedman_mse", "mse", "mae"], default="mse"),
            UniformIntHyperparameter("min_samples_split", 2, 20, default=2),
            UniformIntHyperparameter("min_samples_leaf", 1, 20, default=1),
            UniformFloatHyperparameter("subsample", 0.01, 1.0, default=1.0),
            UniformFloatHyperparameter("max_features", 0.1, 1.0, default=1),
        ],
        constant_hyperparameters={
            "loss": "deviance",
            "min_weight_fraction_leaf": 0,
            "max_leaf_nodes": None,
            "min_impurity_decrease": 0.0,
        })


def random_forest_classifier():
    """Create random forest classifier component."""
    return AlgorithmComponent(
        name="RandomForestClassifier",
        component_class=RandomForestClassifier,
        component_type=AlgorithmType.CLASSIFIER,
        hyperparameters=[
            UniformIntHyperparameter(
                "n_estimators", 50, 200, default=50, n=10),
            CategoricalHyperparameter(
                "criterion", ["gini", "entropy"], default="gini"),
            UniformFloatHyperparameter(
                "max_features", 0.1, 1.0, default=1.0),
            UniformIntHyperparameter(
                "min_samples_split", 2, 20, default=2),
            UniformIntHyperparameter(
                "min_samples_leaf", 1, 20, default=1),
            CategoricalHyperparameter(
                "bootstrap", [True, False], default=True),
            CategoricalHyperparameter(
                "class_weight", ["balanced", None], default=None)
        ],
        constant_hyperparameters={
            "max_depth": None,
            "min_weight_fraction_leaf": 0.0,
            "max_leaf_nodes": None,
            "min_impurity_decrease": 0.0,
        })
