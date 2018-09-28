"""Regressor components."""

from sklearn.linear_model import Lasso, Ridge

from .algorithm import AlgorithmComponent
from .hyperparameter import (
    CategoricalHyperparameter, UniformIntHyperparameter,
    UniformFloatHyperparameter, BaseEstimatorHyperparameter)
from . import constants


# ===================
# Bayesian Regressors
# ===================


# ===================
# Ensemble Regressors
# ===================

def adaboost_regression():
    pass


def extra_trees_regression():
    pass


# ===================
# Gaussian Regressors
# ===================

def rbf_gaussian_process():
    pass


# =================
# Linear Regressors
# =================

def ard_regression():
    pass


def ridge_regression():
    """Create linear ridge regression algorithm component."""
    return AlgorithmComponent(
        name="RidgeRegression",
        component_class=Ridge,
        component_type=constants.REGRESSOR,
        hyperparameters=[
            UniformFloatHyperparameter(
                "alpha", 10 ** -5, 10.0, default=1., log=True),
            UniformFloatHyperparameter(
                "tol", 1e-5, 1e-1, default=1e-3, log=True)
        ],
        constant_hyperparameters={
            "fit_intercept": True,
        })


def lasso_regression():
    """Create linear lasso regression algorithm component."""
    return AlgorithmComponent(
        name="LassoRegression",
        component_class=Lasso,
        component_type=constants.REGRESSOR,
        hyperparameters=[
            UniformFloatHyperparameter(
                "alpha", 10 ** -5, 10.0, default=1., log=True),
            UniformFloatHyperparameter(
                "tol", 1e-5, 1e-1, default=1e-3, log=True)
        ],
        constant_hyperparameters={
            "fit_intercept": True,
        })


def elastic_net_regression():
    pass


def sgd_regression():
    pass


# =========================
# Non-parametric Regressors
# =========================


# =========================
# Support-vector Regressors
# =========================


# =====================
# Tree-based Regressors
# =====================

def decision_tree_regression():
    """Create decision tree regressor algorithm component."""
    pass
