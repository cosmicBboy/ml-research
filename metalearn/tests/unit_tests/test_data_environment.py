"""Unit tests for testing data environment."""

import numpy as np
import pandas as pd

from metalearn import data_types
from metalearn.data_environments import environments, data_environment


def test_envs():
    for dsourcetype in data_types.DataSourceType:
        if dsourcetype != data_types.DataSourceType.SKLEARN:
            continue
        envs = environments.envs(sources=[dsourcetype])
        assert all([isinstance(e.source, data_types.DataSourceType)
                    for e in envs])
        assert all([isinstance(e.target_type, data_types.TargetType)
                    for e in envs])


def test_preprocess_features():
    features = np.array([
        [np.nan, 2.5, 0, "foo", np.datetime64("1999-07-17")],
        [1.5, np.nan, 1, "bar", np.datetime64("2008-05-09")],
        [1.8, 4.2, np.nan, "foo", np.datetime64("2011-01-18")],
        [1.1, 6.0, np.nan, np.nan, np.datetime64("2015-02-01")]],
        # preserve datatypes of each column in the array
        dtype="O")
    feature_types = [
        data_types.FeatureType.CONTINUOUS,
        data_types.FeatureType.CONTINUOUS,
        data_types.FeatureType.CATEGORICAL,
        data_types.FeatureType.CATEGORICAL,
        data_types.FeatureType.DATE,
    ]

    # categorical features should be stacked to the right
    expected_array = [
        [np.nan, 2.5, 1999, 7, 198, 17, 5, 0, "foo"],
        [1.5, np.nan, 2008, 5, 130, 9, 4, 1, "bar"],
        [1.8, 4.2, 2011, 1, 18, 18, 1, np.nan, "foo"],
        [1.1, 6.0, 2015, 2, 32, 1, 6, np.nan, np.nan],
    ]
    expected_feature_features = [
        data_types.FeatureType.CONTINUOUS,
        data_types.FeatureType.CONTINUOUS,
        # date variables are converted into 5 continuous
        data_types.FeatureType.CONTINUOUS,
        data_types.FeatureType.CONTINUOUS,
        data_types.FeatureType.CONTINUOUS,
        data_types.FeatureType.CONTINUOUS,
        data_types.FeatureType.CONTINUOUS,
        data_types.FeatureType.CATEGORICAL,
        data_types.FeatureType.CATEGORICAL,
    ]
    expected_feature_indices = list(range(len(expected_feature_features)))

    preprocessed_features = data_environment.preprocess_features(
        features, feature_types)
    result_array = [list(x) for x in list(preprocessed_features.X)]

    assert pd.DataFrame(expected_array).equals(pd.DataFrame(result_array))
    assert expected_feature_features == preprocessed_features.feature_types
    assert expected_feature_indices == preprocessed_features.feature_indices
    assert preprocessed_features.feature_names is None

    # preprocess features with feature_names
    feature_names = [
        "continuous1",
        "continuous2",
        "categorical1",
        "categorical2",
        "date",
    ]
    expected_feature_names = [
        "continuous1",
        "continuous2",
        "date_year",
        "date_month",
        "date_day_of_year",
        "date_day",
        "date_day_of_week",
        "categorical1",
        "categorical2",
    ]

    preprocessed_features = data_environment.preprocess_features(
        features, feature_types, feature_names)
    result_array = [list(x) for x in list(preprocessed_features.X)]

    assert pd.DataFrame(expected_array).equals(pd.DataFrame(result_array))
    assert expected_feature_features == preprocessed_features.feature_types
    assert expected_feature_indices == preprocessed_features.feature_indices
    assert expected_feature_names == preprocessed_features.feature_names
