"""Regressor components."""

from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import ARDRegression, BayesianRidge, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVR, SVR
from sklearn.tree import DecisionTreeRegressor

from .algorithm_component import AlgorithmComponent, EXCLUDE_ALL
from .hyperparameter import (
    CategoricalHyperparameter, UniformIntHyperparameter,
    UniformFloatHyperparameter, BaseEstimatorHyperparameter)
from ..data_types import AlgorithmType


# TODO: add sgd regressor, xgboost regressor


# ===================
# Ensemble Regressors
# ===================

def adaboost_regression():
    return AlgorithmComponent(
        name="AdaBoostRegressor",
        component_class=AdaBoostRegressor,
        component_type=AlgorithmType.REGRESSOR,
        hyperparameters=[
            BaseEstimatorHyperparameter(
                hname="base_estimator",
                base_estimator=DecisionTreeRegressor,
                hyperparameters=[
                    UniformIntHyperparameter("max_depth", 1, 10, default=1)
                ],
                default=DecisionTreeRegressor(max_depth=1)
            ),
            UniformIntHyperparameter(
                "n_estimators", 50, 500, default=50, n=10),
            UniformFloatHyperparameter(
                "learning_rate", 0.1, 2., default=1., log=True, n=10),
            CategoricalHyperparameter(
                "loss", ["linear", "square", "exponential"], default="linear")
        ])


def extra_trees_regression():
    pass


def gradient_boosting_regression():
    pass


def random_forest_regression():
    return AlgorithmComponent(
        name="RandomForestRegressor",
        component_class=RandomForestRegressor,
        component_type=AlgorithmType.REGRESSOR,
        hyperparameters=[
            UniformIntHyperparameter(
                "n_estimators", 50, 200, default=50, n=10),
            CategoricalHyperparameter(
                "criterion", ["mse", "friedman_mse", "mae"], default="mse"),
            UniformFloatHyperparameter(
                "max_features", 0.1, 1.0, default=1.0),
            UniformIntHyperparameter(
                "min_samples_split", 2, 20, default=2),
            UniformIntHyperparameter(
                "min_samples_leaf", 1, 20, default=1),
            CategoricalHyperparameter(
                "bootstrap", [True, False], default=True),
        ],
        constant_hyperparameters={
            "max_depth": None,
            "min_weight_fraction_leaf": 0.0,
            "max_leaf_nodes": None,
            "min_impurity_decrease": 0.0,
        })


# ===================
# Gaussian Regressors
# ===================

def rbf_gaussian_process_regression():
    """Create gaussian process regressorw with an RBF kernel.

    Use the default "1.0 * RBF(1.0)" kernel.

    Auto-sklearn implements the kernel as dependent on `n_features` of the
    training data:
    https://github.com/automl/auto-sklearn/blob/master/autosklearn/pipeline/components/regression/gaussian_process.py  # noqa

    Later need to decide whether it's worth supporting this kind of
    implementation by dynamically instantiating the estimator based on the
    data environment.
    """
    return AlgorithmComponent(
        name="GaussianProcessRegressor",
        component_class=GaussianProcessRegressor,
        component_type=AlgorithmType.REGRESSOR,
        hyperparameters=[
            UniformFloatHyperparameter(
                "alpha", 1e-14, 1.0, default=1e-8, log=True),
        ],
        constant_hyperparameters={
            "optimizer": "fmin_l_bfgs_b",
            "n_restarts_optimizer": 10,
            "normalize_y": True,
        })


# =================
# Linear Regressors
# =================

def ard_regression():
    return AlgorithmComponent(
        name="ARDRegression",
        component_class=ARDRegression,
        component_type=AlgorithmType.REGRESSOR,
        hyperparameters=[
            UniformFloatHyperparameter(
                "tol", 1e-5, 1e-1, default=1e-3, log=True),
            UniformFloatHyperparameter(
                "alpha_1", 1e-10, 1e-3, default=1e-6, log=True),
            UniformFloatHyperparameter(
                "alpha_2", 1e-10, 1e-3, default=1e-6, log=True),
            UniformFloatHyperparameter(
                "lambda_1", 1e-10, 1e-3, default=1e-6, log=True),
            UniformFloatHyperparameter(
                "lambda_2", 1e-10, 1e-3, default=1e-6, log=True),
            UniformFloatHyperparameter(
                "threshold_lambda", 10 ** 3, 10 ** 5, default=10 ** 4,
                log=True),
        ],
        constant_hyperparameters={
            "n_iter": 300,
            "fit_intercept": True,
            "compute_score": False
        })


def bayesian_ridge_regression():
    return AlgorithmComponent(
        name="BayesianRidge",
        component_class=BayesianRidge,
        component_type=AlgorithmType.REGRESSOR,
        hyperparameters=[
            UniformFloatHyperparameter(
                "tol", 1e-5, 1e-1, default=1e-3, log=True),
            UniformFloatHyperparameter(
                "alpha_1", 1e-10, 1e-3, default=1e-6, log=True),
            UniformFloatHyperparameter(
                "alpha_2", 1e-10, 1e-3, default=1e-6, log=True),
            UniformFloatHyperparameter(
                "lambda_1", 1e-10, 1e-3, default=1e-6, log=True),
            UniformFloatHyperparameter(
                "lambda_2", 1e-10, 1e-3, default=1e-6, log=True),
        ],
        constant_hyperparameters={
            "n_iter": 300,
            "fit_intercept": True,
            "compute_score": False
        })


def ridge_regression():
    """Create linear ridge regression algorithm component."""
    return AlgorithmComponent(
        name="RidgeRegression",
        component_class=Ridge,
        component_type=AlgorithmType.REGRESSOR,
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
        component_type=AlgorithmType.REGRESSOR,
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

def k_nearest_neighbors_regression():
    return AlgorithmComponent(
        name="KNearestNeighorsRegressor",
        component_class=KNeighborsRegressor,
        component_type=AlgorithmType.REGRESSOR,
        hyperparameters=[
            UniformIntHyperparameter(
                "n_neighbors", 1, 100, log=True, default=1),
            CategoricalHyperparameter(
                "weights", ["uniform", "distance"], default="uniform"),
            CategoricalHyperparameter("p", [1, 2], default=2)
        ])


# =========================
# Support-vector Regressors
# =========================


def support_vector_regression_linear():
    # just following the parameterization used by auto-sklearn:
    # https://github.com/automl/auto-sklearn/blob/master/autosklearn/pipeline/components/regression/liblinear_svr.py  # noqa
    return AlgorithmComponent(
        name="LinearSVR",
        component_class=LinearSVR,
        component_type=AlgorithmType.REGRESSOR,
        hyperparameters=[
            UniformFloatHyperparameter(
                "C", 0.03125, 32768, default=1.0, log=True, n=10),
            UniformFloatHyperparameter(
                "epsilon", 0.001, 1, default=0.1, log=True, n=10),
            UniformFloatHyperparameter(
                "tol", 1e-5, 1e-1, default=1e-4, log=True, n=10),
        ],
        constant_hyperparameters={
            "loss": "squared_epsilon_insensitive",
            "dual": False,
            "fit_intercept": True,
            "intercept_scaling": 1,
        })


def support_vector_regression_nonlinear():
    return AlgorithmComponent(
        name="NonlinearSVR",
        component_class=SVR,
        component_type=AlgorithmType.REGRESSOR,
        hyperparameters=[
            UniformFloatHyperparameter(
                "C", 0.03125, 32768, default=1.0, log=True),
            UniformFloatHyperparameter(
                "epsilon", 0.001, 1, default=0.1, log=True),
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
        },
        exclusion_conditions={
            "kernel": {
                "rbf": {
                    "coef0": EXCLUDE_ALL,
                    "degree": EXCLUDE_ALL,
                },
                "sigmoid": {
                    "degree": EXCLUDE_ALL
                }
            }
        })


def _libsvm_hyperparameters():
    return [
        UniformFloatHyperparameter(
            "C", 0.03125, 32768, default=1.0, log=True),
        UniformFloatHyperparameter(
            "epsilon", 0.001, 1, default=0.1, log=True),
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
    }


def support_vector_regression_poly():
    constant_hyperparameters = {"kernel": "poly"}
    constant_hyperparameters.update(_libsvm_constant_hyperparameters())
    return AlgorithmComponent(
        name="PolyKernelSVR",
        component_class=SVR,
        component_type=AlgorithmType.REGRESSOR,
        hyperparameters=(
            _libsvm_hyperparameters() +
            _libsvm_poly_sigmoid_hyperparameters() +
            [UniformIntHyperparameter("degree", 2, 5, default=3)]
        ),
        constant_hyperparameters=constant_hyperparameters,
    )


def support_vector_regression_rbf():
    constant_hyperparameters = {"kernel": "rbf"}
    constant_hyperparameters.update(_libsvm_constant_hyperparameters())
    return AlgorithmComponent(
        name="RBFKernelSVR",
        component_class=SVR,
        component_type=AlgorithmType.REGRESSOR,
        hyperparameters=_libsvm_hyperparameters(),
        constant_hyperparameters=constant_hyperparameters
    )


def support_vector_regression_sigmoid():
    constant_hyperparameters = {"kernel": "sigmoid"}
    constant_hyperparameters.update(_libsvm_constant_hyperparameters())
    return AlgorithmComponent(
        name="SigmoidKernelSVR",
        component_class=SVR,
        component_type=AlgorithmType.REGRESSOR,
        hyperparameters=(
            _libsvm_hyperparameters() +
            _libsvm_poly_sigmoid_hyperparameters()
        ),
        constant_hyperparameters=constant_hyperparameters,
    )


# =====================
# Tree-based Regressors
# =====================

def decision_tree_regression():
    """Create decision tree regressor algorithm component."""
    return AlgorithmComponent(
        name="DecisionTreeRegressor",
        component_class=DecisionTreeRegressor,
        component_type=AlgorithmType.REGRESSOR,
        hyperparameters=[
            CategoricalHyperparameter(
                "criterion", ["mse", "friedman_mse", "mae"], default="mse"),
            CategoricalHyperparameter(
                "splitter", ["best", "random"], default="best"),
            UniformIntHyperparameter("max_depth", 1, 4, default=1, n=4),
            UniformIntHyperparameter(
                "min_samples_split", 2, 20, default=2, n=10),
            UniformIntHyperparameter(
                "min_samples_leaf", 1, 20, default=1, n=10),
        ],
        constant_hyperparameters={
            "min_weight_fraction_leaf": 0.0,
            "max_features": 1,
            "max_leaf_nodes": None,
            "min_impurity_decrease": 0.0,
        })
