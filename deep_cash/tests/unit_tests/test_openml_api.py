"""Unit and integration tests for Openml API."""

from deep_cash.data_types import TargetType
from deep_cash.data_environments import openml_api


def test_classification_envs():
    datasets = openml_api.classification_envs(n=10)
    # datasets can be invalid for several reasons (no valid target column,
    # no features parsed)
    assert len(datasets) <= 10


def test_regression_envs():
    reg_envs = openml_api.regression_envs(3).values()
    assert len(reg_envs) == 3
    assert all([
        e.target_type == TargetType.REGRESSION for e in reg_envs])
