"""Unit tests for regressor algorithm components."""

import sklearn

from metalearn.components import regressors
from sklearn.base import BaseEstimator


REGRESSORS = [
    regressors.adaboost_regression,
    regressors.ard_regression,
    regressors.bayesian_ridge_regression,
    regressors.decision_tree_regression,
    regressors.random_forest_regression,
    regressors.rbf_gaussian_process_regression,
    regressors.ridge_regression,
    regressors.lasso_regression,
    regressors.k_nearest_neighbors_regression,
    regressors.support_vector_regression_linear,
    regressors.support_vector_regression_poly,
    regressors.support_vector_regression_rbf,
    regressors.support_vector_regression_sigmoid,
]


def test_regressor_components():
    """Ensure that a classifier can be fitted and used to make predictions."""
    boston = sklearn.datasets.load_boston()
    for algorithm_component in REGRESSORS:
        reg_component = algorithm_component()

        hyperparam_name_space = reg_component.hyperparameter_name_space()
        if hyperparam_name_space is None:
            continue
        for hname in hyperparam_name_space:
            assert hname.startswith(reg_component.name)

        X, y = boston.data[:20], boston.target[:20]

        reg = reg_component()
        assert isinstance(reg, BaseEstimator)
        reg.fit(X, y)
        y_hat = reg.predict(X)
        mse = sklearn.metrics.mean_squared_error(y, y_hat)
        assert 0 <= mse


def test_regressor_components():

    for algorithm_component in REGRESSORS:
        reg_component = algorithm_component()
        estimator = reg_component()
        for _ in range(20):
            hyperparams = {
                k.split("__")[1]: v for k, v in
                reg_component.sample_hyperparameter_state_space().items()
            }
            assert isinstance(
                estimator.set_params(**hyperparams), BaseEstimator)
