"""Unit tests for data preprocessor algorithm components."""

import sklearn

from metalearn.components import data_preprocessors
from sklearn.base import BaseEstimator, TransformerMixin


DATA_PREPROCESSORS = [
    data_preprocessors.simple_imputer,
    data_preprocessors.one_hot_encoder,
    data_preprocessors.minmax_scaler,
    data_preprocessors.standard_scaler,
    data_preprocessors.robust_scaler,
    data_preprocessors.normalizer,
]


def test_data_preprocessor_components():
    """Ensure that a classifier can be fitted and used to make predictions."""
    iris = sklearn.datasets.load_iris()
    iris_task_metadata = {
        "continuous_features": list(range(4)),
        "categorical_features": []
    }

    boston = sklearn.datasets.load_boston()
    boston_task_metadata = {
        "continuous_features": [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "categorical_features": [3],
    }
    for algorithm_component in DATA_PREPROCESSORS:
        dpp_component = algorithm_component()

        hyperparam_name_space = dpp_component.hyperparameter_name_space()
        if hyperparam_name_space is None:
            continue
        for hname in hyperparam_name_space:
            assert hname.startswith(dpp_component.name)

        dpp_iris = dpp_component(**iris_task_metadata)
        dpp_iris.set_params(**dpp_component.constant_hyperparameters)
        assert isinstance(dpp_iris, BaseEstimator)
        assert iris.data.shape[0] == \
            dpp_iris.fit_transform(iris.data, iris.target).shape[0]

        dpp_boston = dpp_component(**boston_task_metadata)
        dpp_boston.set_params(**dpp_component.constant_hyperparameters)
        assert isinstance(dpp_boston, BaseEstimator)
        assert boston.data.shape[0] == \
            dpp_boston.fit_transform(boston.data, boston.target).shape[0]


def test_classifier_set_params():
    """Test that hyperparameters can be set on objects produced by __call__."""

    for algorithm_component in DATA_PREPROCESSORS:
        dpp_component = algorithm_component()
        estimator = dpp_component()
        for _ in range(20):
            hyperparams = {
                k.split("__")[1]: v for k, v in
                dpp_component.sample_hyperparameter_state_space().items()
            }
            assert isinstance(
                estimator.set_params(**hyperparams),
                (BaseEstimator, TransformerMixin))
