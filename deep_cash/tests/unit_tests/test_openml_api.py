"""Unit and integration tests for Openml API."""

from deep_cash.data_types import TargetType
from deep_cash.data_environments import openml_api


def test_list_clf_datasets():
    dataset_metadata = openml_api.list_clf_datasets(n_results=10)
    assert len(dataset_metadata) == 10
    for metadata in dataset_metadata.values():
        assert metadata["NumberOfMissingValues"] == 0
        assert openml_api.CLASS_RANGE[0] <= metadata["NumberOfClasses"] <= \
            openml_api.CLASS_RANGE[1]


def test_classification_envs():
    datasets = openml_api.classification_envs(n=10)
    # datasets can be invalid for several reasons (no valid target column,
    # no features parsed)
    assert len(datasets) <= 10


def test_regression_envs():
    reg_envs = openml_api.regression_envs(3)
    assert len(reg_envs) == 3
    assert all([
        e["target_type"] == TargetType.REGRESSION for e in reg_envs])
