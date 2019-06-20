"""Module for sampling from a distribution of classification environments."""

from collections import OrderedDict
from functools import partial

import sklearn.datasets
import sklearn.preprocessing

from .data_environment import DataEnvironment
from ..data_types import FeatureType, TargetType, DataSourceType


def envs(n=5, test_size=None, random_state=None, verbose=False):
    # TODO: limit number of envs by n
    return OrderedDict([
        ("sklearn.iris", DataEnvironment(
            name="sklearn.iris",
            source=DataSourceType.SKLEARN,
            target_type=TargetType.MULTICLASS,
            feature_types=[FeatureType.CONTINUOUS for _ in range(4)],
            feature_indices=range(4),
            fetch_training_data=partial(
                sklearn.datasets.load_iris, return_X_y=True),
            fetch_test_data=None,
            test_size=test_size,
            random_state=random_state,
            target_preprocessor=None,
            scorer=None)),
        ("sklearn.digits", DataEnvironment(
            name="sklearn.digits",
            source=DataSourceType.SKLEARN,
            target_type=TargetType.MULTICLASS,
            feature_types=[FeatureType.CONTINUOUS for _ in range(64)],
            feature_indices=range(64),
            fetch_training_data=partial(
                sklearn.datasets.load_digits, return_X_y=True),
            fetch_test_data=None,
            test_size=test_size,
            random_state=random_state,
            target_preprocessor=None,
            scorer=None)),
        ("sklearn.wine", DataEnvironment(
            name="sklearn.wine",
            source=DataSourceType.SKLEARN,
            target_type=TargetType.MULTICLASS,
            feature_types=[FeatureType.CONTINUOUS for _ in range(13)],
            feature_indices=range(13),
            fetch_training_data=partial(
                sklearn.datasets.load_wine, return_X_y=True),
            fetch_test_data=None,
            test_size=test_size,
            random_state=random_state,
            target_preprocessor=None,
            scorer=None)),
        ("sklearn.breast_cancer", DataEnvironment(
            name="sklearn.breast_cancer",
            source=DataSourceType.SKLEARN,
            target_type=TargetType.BINARY,
            feature_types=[FeatureType.CONTINUOUS for _ in range(30)],
            feature_indices=range(39),
            fetch_training_data=partial(
                sklearn.datasets.load_breast_cancer, return_X_y=True),
            fetch_test_data=None,
            test_size=test_size,
            random_state=random_state,
            target_preprocessor=None,
            scorer=None)),
    ])
