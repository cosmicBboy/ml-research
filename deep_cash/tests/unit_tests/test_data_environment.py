"""Unit tests for testing data environment."""

import numpy as np

from deep_cash import data_types
from deep_cash.data_environments import environments, data_environment


def test_envs():
    for dsourcetype in data_types.DataSourceType:
        envs = environments.envs(sources=[dsourcetype])
        assert all([isinstance(e.source, data_types.DataSourceType)
                    for e in envs])
        assert all([isinstance(e.target_type, data_types.TargetType)
                    for e in envs])


def test_preprocess_features():
    features = np.array([
        [np.nan, 2.5, 0, np.datetime64("1999-07-17")],
        [1.5, np.nan, 1, np.datetime64("2008-05-09")],
        [1.8, 4.2, np.nan, np.datetime64("2011-01-18")],
        [1.1, 6.0, np.nan, np.datetime64("2015-02-01")]],
        # preserve datatypes of each column in the array
        dtype="O")
    feature_types = [
        data_types.FeatureType.CONTINUOUS,
        data_types.FeatureType.CONTINUOUS,
        data_types.FeatureType.CATEGORICAL,
        data_types.FeatureType.DATE,
    ]
    expected = np.array([
        [np.nan, 2.5, 0, 1999, 7, 198, 17, 5],
        [1.5, np.nan, 1, 2008, 5, 130, 9, 4],
        [1.8, 4.2, 2, 2011, 1, 18, 18, 1],
        [1.1, 6.0, 2, 2015, 2, 32, 1, 6],
    ])
    result = data_environment.preprocess_features(features, feature_types)
    assert np.allclose(result, expected, equal_nan=True)


def test_handle_missing_categorical():
    a1 = np.array([1, 2, 3, 4])
    a2 = np.array([1, 5, 10, 12, 11])
    a3 = np.array([1.1, 2.2, 4.3, 2.2])
    a4 = np.array([np.nan, 1, np.nan, 2, 3])
    a5 = np.array([np.nan, "a", "b", np.nan, "c"], dtype="O")

    assert (data_environment.handle_missing_categorical(a1) ==
            np.array([0, 1, 2, 3])).all()
    assert (data_environment.handle_missing_categorical(a2) ==
            np.array([0, 1, 2, 4, 3])).all()
    assert (data_environment.handle_missing_categorical(a3) ==
            np.array([0, 1, 2, 1])).all()
    assert (data_environment.handle_missing_categorical(a4) ==
            np.array([3, 0, 3, 1, 2])).all()
    assert (data_environment.handle_missing_categorical(a5) ==
            np.array([3, 0, 1, 3, 2])).all()
