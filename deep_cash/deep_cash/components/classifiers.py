"""Classifier components."""

from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier

from .algorithm import AlgorithmComponent
from .hyperparameter import (
    CategoricalHyperparameter, UniformIntHyperparameter,
    UniformFloatHyperparameter, BaseEstimatorHyperparameter)
from . import constants


# ====================
# Gaussian Classifiers
# ====================

def gaussian_naive_bayes():
    """Create a naive bayes algorithm component."""
    return AlgorithmComponent(
        name="GaussianNaiveBayes",
        component_class=GaussianNB,
        component_type=constants.CLASSIFIER)


def rbf_gaussian_process():
    pass


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
        component_type=constants.CLASSIFIER,
        hyperparameters=[
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
        component_type=constants.CLASSIFIER,
        hyperparameters=[
            UniformIntHyperparameter(
                "n_neighbors", 1, 100, log=True, default=1),
            CategoricalHyperparameter(
                "weights", ["uniform", "distance"], default="uniform"),
        ])


# ==========================
# Support-vector Classifiers
# ==========================
#
# NOTE: supporting conditional hyperparameters would introduce more complexity
# to the cash controller decoder implementation, which is more trouble than
# it's worth for now. This is because the controller only chooses
# hyperparameter values in an array of hyperparameters for a particular
# estimator/transformer.
#
# In order to support conditional hyperparameters, the controller would need
# to keep track of conditional dependencies among softmax (hyperparameter
# action) classifiers, i.e. if "rbf" kernel is picked, then the "degree"
# hyperparameter shouldn't be chosen.
#
# Options:
# 1) just specify all hyperparameters in a single algorithm component,
#    with the assumption that sklearn will handle ignoring irrelevant
#    hyperparameters (and also eat the added computational cost of spending
#    training time on hyperparameter settings that don't make sense).
# 2) express the conditionality of hyperparameters by spinning off different
#    algorithm components.
# 3) support conditional hyperparameters in the CashController.
#
# Going for option (2) for now, gonna wait and see if there are more
# justifications to go for (3) as the project goes on.

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
        component_type=constants.CLASSIFIER,
        hyperparameters=[
            CategoricalHyperparameter("penalty", ["l1", "l2"], default="l2"),
            CategoricalHyperparameter(
                "loss", ["hinge", "squared_hinge"], default="squared_hinge"),
            UniformFloatHyperparameter(
                "tol", 1e-5, 1e-1, default=1e-4, log=True),
            UniformFloatHyperparameter(
                "C", 0.03215, 32768, log=True, default=1.0)
        ],
        constant_hyperparameters={
            "dual": False,
            "multi_class": "ovr",
        })


def _libsvm_hyperparameters():
    return [
        UniformFloatHyperparameter(
            "C", 0.03125, 32768, default=1.0, log=True),
        UniformFloatHyperparameter(
            "gamma", 3.0517578125e-05, 8, default=0.1, log=True),
        CategoricalHyperparameter(
            "shrinking", [True, False], default=True),
        UniformFloatHyperparameter(
            "tol", 1e-5, 1e-1, default=1e-3, log=True),
    ]


def _libsvm_poly_sigmoid_hyperparameters():
    return [
        UniformFloatHyperparameter("coef0", -1, -1, default=0)
    ]


def _libsvm_constant_hyperparameters():
    return {
        "max_iter": -1,
        "cache_size": 200,
        "decision_function_shape": "ovr",
    }


def support_vector_classifier_poly():
    """Create linear support vector classifier component."""
    constant_hyperparameters = {"kernel": "poly"}
    constant_hyperparameters.update(_libsvm_constant_hyperparameters())
    return AlgorithmComponent(
        name="PolyKernelSVC",
        component_class=SVC,
        component_type=constants.CLASSIFIER,
        hyperparameters=(
            _libsvm_hyperparameters() +
            _libsvm_poly_sigmoid_hyperparameters() +
            [UniformIntHyperparameter("degree", 2, 5, default=3)]
        ),
        constant_hyperparameters=constant_hyperparameters,
    )


def support_vector_classifier_rbf():
    """Create linear support vector classifier component."""
    constant_hyperparameters = {"kernel": "rbf"}
    constant_hyperparameters.update(_libsvm_constant_hyperparameters())
    return AlgorithmComponent(
        name="RBFKernelSVC",
        component_class=SVC,
        component_type=constants.CLASSIFIER,
        hyperparameters=_libsvm_hyperparameters(),
        constant_hyperparameters=constant_hyperparameters
    )


def support_vector_classifier_sigmoid():
    """Create linear support vector classifier component."""
    constant_hyperparameters = {"kernel": "sigmoid"}
    constant_hyperparameters.update(_libsvm_constant_hyperparameters())
    return AlgorithmComponent(
        name="SigmoidKernelSVC",
        component_class=SVC,
        component_type=constants.CLASSIFIER,
        hyperparameters=(
            _libsvm_hyperparameters() +
            _libsvm_poly_sigmoid_hyperparameters()
        ),
        constant_hyperparameters=constant_hyperparameters,
    )


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
        component_type=constants.CLASSIFIER,
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
        component_type=constants.CLASSIFIER,
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


def bagging():
    """Create bagging classifier component."""
    pass


def extra_trees():
    """Create extra trees classifier component."""
    pass


def gradient_boosting():
    """Create gradient boosting classifier component."""
    pass


def random_forest():
    """Create random forest classifier component."""
    pass
