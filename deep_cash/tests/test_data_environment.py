"""Unit tests for testing data environment."""

import numpy as np

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


def test_preprocess_data_env():
    data_env = {
        "dataset_name": "some_env",
        "target_type": data_types.TargetType.BINARY,
        "data": np.array([
            [np.nan, 2.5, 0],
            [1.5, np.nan, 1],
            [1.8, 4.2, np.nan],
            [1.1, 6.0, np.nan]
        ]),
        "target": np.array([0, 1, 0, 1]),
        "feature_types": [
            data_types.FeatureType.CONTINUOUS,
            data_types.FeatureType.CONTINUOUS,
            data_types.FeatureType.CATEGORICAL,
        ],
        "feature_indices": [0, 1, 2, 3],
        "target_preprocessor": None,
        "source": "some_source"
    }
    expected = np.array([
        [np.nan, 2.5, 0],
        [1.5, np.nan, 1],
        [1.8, 4.2, 2],
        [1.1, 6.0, 2]
    ])
    result = environments.preprocess_data_env(data_env)
    assert np.allclose(result["data"], expected, equal_nan=True)


def test_handle_missing_categorical():
    a1 = np.array([1, 2, 3, 4])
    a2 = np.array([1, 5, 10, 12, 11])
    a3 = np.array([1.1, 2.2, 4.3, 2.2])
    a4 = np.array([np.nan, 1, np.nan, 2, 3])

    assert (environments.handle_missing_categorical(a1) ==
            np.array([0, 1, 2, 3])).all()
    assert (environments.handle_missing_categorical(a2) ==
            np.array([0, 1, 2, 4, 3])).all()
    assert (environments.handle_missing_categorical(a3) ==
            np.array([0, 1, 2, 1])).all()
    assert (environments.handle_missing_categorical(a4) ==
            np.array([3, 0, 3, 1, 2])).all()
