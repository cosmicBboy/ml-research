"""Unit and integration tests for Openml API."""

from metalearn.data_types import TargetType
from metalearn.data_environments import openml_api
from metalearn.data_environments.data_environment import DataEnvironment


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


def test_autosklearn_paper_classification_envs():
    autosklearn_envs = openml_api.autosklearn_paper_classification_envs(n=10)
    assert all([
        isinstance(x, DataEnvironment) for x in autosklearn_envs
    ])
