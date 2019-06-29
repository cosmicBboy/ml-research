"""Unit tests for feature preprocessor algorithm components."""

import sklearn

from metalearn.components import feature_preprocessors
from sklearn.base import BaseEstimator, TransformerMixin


FEATURE_PREPROCESSORS = [
    feature_preprocessors.fast_ica,
    feature_preprocessors.feature_agglomeration,
    feature_preprocessors.kernel_pca,
    feature_preprocessors.rbf_sampler,
    feature_preprocessors.nystroem_sampler,
    feature_preprocessors.pca,
    feature_preprocessors.polynomial_features,
    feature_preprocessors.random_trees_embedding,
    feature_preprocessors.truncated_svd,
    feature_preprocessors.variance_threshold_filter,
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
    for algorithm_component in FEATURE_PREPROCESSORS:
        fpp_component = algorithm_component()

        hyperparam_name_space = fpp_component.hyperparameter_name_space()
        if hyperparam_name_space is None:
            continue
        for hname in hyperparam_name_space:
            assert hname.startswith(fpp_component.name)

        fpp_iris = fpp_component(**iris_task_metadata)
        fpp_iris.set_params(**fpp_component.constant_hyperparameters)
        assert isinstance(fpp_iris, BaseEstimator)
        assert iris.data.shape[0] == \
            fpp_iris.fit_transform(iris.data, iris.target).shape[0]

        fpp_boston = fpp_component(**boston_task_metadata)
        fpp_boston.set_params(**fpp_component.constant_hyperparameters)
        assert isinstance(fpp_boston, BaseEstimator)
        assert boston.data.shape[0] == \
            fpp_boston.fit_transform(boston.data, boston.target).shape[0]


def test_classifier_set_params():
    """Test that hyperparameters can be set on objects produced by __call__."""

    for algorithm_component in FEATURE_PREPROCESSORS:
        fpp_component = algorithm_component()
        estimator = fpp_component()
        for _ in range(20):
            hyperparams = {
                k.split("__")[1]: v for k, v in
                fpp_component.sample_hyperparameter_state_space().items()
            }
            assert isinstance(
                estimator.set_params(**hyperparams),
                (BaseEstimator, TransformerMixin))
