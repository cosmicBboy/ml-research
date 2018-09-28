"""Unit tests for data env."""

from deep_cash.data_types import TargetType
from deep_cash.data_sourcers import openml_api


def test_regression_envs():
    reg_envs = openml_api.regression_envs(3)
    assert len(reg_envs) == 3
    assert all([
        e["target_type"] == TargetType.REGRESSION for e in reg_envs])
