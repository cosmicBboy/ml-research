"""Module for sampling from a distribution of classification environments."""

from collections import OrderedDict

import sklearn.datasets
import sklearn.preprocessing

from ..data_types import FeatureType, TargetType, DataSourceType


SKLEARN_DATA_ENV_CONFIG = OrderedDict([
    (sklearn.datasets.load_iris, {
        "dataset_name": "iris",
        "target_type": TargetType.MULTICLASS,
        "feature_types": [FeatureType.CONTINUOUS for _ in range(4)],
        "feature_indices": range(4),
        "target_preprocessor": None,
        "source": DataSourceType.SKLEARN}),
    (sklearn.datasets.load_digits, {
        "dataset_name": "digits",
        "target_type": TargetType.MULTICLASS,
        "feature_types": [FeatureType.CONTINUOUS for _ in range(64)],
        "feature_indices": range(64),
        "target_preprocessor": None,
        "source": DataSourceType.SKLEARN}),
    (sklearn.datasets.load_wine, {
        "dataset_name": "wine",
        "target_type": TargetType.MULTICLASS,
        "feature_types": [FeatureType.CONTINUOUS for _ in range(13)],
        "feature_indices": range(13),
        "target_preprocessor": None,
        "source": DataSourceType.SKLEARN}),
    (sklearn.datasets.load_breast_cancer, {
        "dataset_name": "breast_cancer",
        "target_type": TargetType.BINARY,
        "feature_types": [FeatureType.CONTINUOUS for _ in range(30)],
        "feature_indices": range(30),
        "target_preprocessor": None,
        "source": DataSourceType.SKLEARN,
        }),
])
