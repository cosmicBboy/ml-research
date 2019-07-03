"""Module for sampling from a distribution of regression environments."""

from collections import OrderedDict
from functools import partial

import sklearn.datasets

from .data_environment import DataEnvironment
from ..data_types import FeatureType, TargetType, DataSourceType


def envs(n=5, test_size=None, random_state=None, verbose=False):
    # TODO: limit number of envs by n
    return OrderedDict([
        ("sklearn.boston", DataEnvironment(
            name="sklearn.boston",
            source=DataSourceType.SKLEARN,
            target_type=TargetType.REGRESSION,
            feature_types=[
                FeatureType.CONTINUOUS,  # CRIM
                FeatureType.CONTINUOUS,  # ZN
                FeatureType.CONTINUOUS,  # INDUS
                FeatureType.CATEGORICAL,  # CHAS
                FeatureType.CONTINUOUS,  # NOX
                FeatureType.CONTINUOUS,  # RM
                FeatureType.CONTINUOUS,  # AGE
                FeatureType.CONTINUOUS,  # DIS
                FeatureType.CONTINUOUS,  # RAD
                FeatureType.CONTINUOUS,  # TAX
                FeatureType.CONTINUOUS,  # PTRATIO
                FeatureType.CONTINUOUS,  # B
                FeatureType.CONTINUOUS,  # LSTAT
            ],
            feature_indices=range(13),
            fetch_training_data=partial(
                sklearn.datasets.load_boston, return_X_y=True),
            fetch_test_data=None,
            test_size=test_size,
            random_state=random_state,
            target_preprocessor=None,
            scorer=None)),
        ("sklearn.diabetes", DataEnvironment(
            name="sklearn.diabetes",
            source=DataSourceType.SKLEARN,
            target_type=TargetType.REGRESSION,
            feature_types=[FeatureType.CONTINUOUS for _ in range(10)],
            feature_indices=range(10),
            fetch_training_data=partial(
                sklearn.datasets.load_diabetes, return_X_y=True),
            fetch_test_data=None,
            test_size=test_size,
            random_state=random_state,
            target_preprocessor=None,
            scorer=None)),
        ("sklearn.linnerud", DataEnvironment(
            name="sklearn.linnerud",
            source=DataSourceType.SKLEARN,
            target_type=TargetType.MULTIREGRESSION,
            feature_types=[FeatureType.CONTINUOUS for _ in range(3)],
            feature_indices=range(3),
            fetch_training_data=partial(
                sklearn.datasets.load_linnerud, return_X_y=True),
            fetch_test_data=None,
            test_size=test_size,
            random_state=random_state,
            target_preprocessor=None,
            scorer=None)),
    ])
