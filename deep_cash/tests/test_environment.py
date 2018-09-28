"""Unit tests for testing data environment."""

from deep_cash import data_types
from deep_cash.data_environments import environments


def test_envs():
    all_envs = environments.envs()
    assert all([isinstance(e["target_type"], data_types.TargetType)
                for e in all_envs])

    for dsourcetype in data_types.DataSourceType:
        envs = environments.envs(sources=[dsourcetype])
        assert all([isinstance(e["source"], data_types.DataSourceType)
                    for e in envs])
