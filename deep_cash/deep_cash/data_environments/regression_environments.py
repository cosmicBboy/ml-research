"""Module for sampling from a distribution of regression environments."""

from collections import OrderedDict

import sklearn.datasets

from ..data_types import FeatureType, TargetType


# see dataset.DESCR for more on the feature set.
SKLEARN_DATA_ENV_CONFIG = OrderedDict([
    (sklearn.datasets.load_boston, {
        "dataset_name": "boston",
        "target_type": TargetType.REGRESSION,
        "feature_types": [
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
            FeatureType.CONTINUOUS,  # MEDV
        ],
        "feature_indices": range(14),
        "target_preprocessor": None}),
])
