"""Module for sampling from a distribution of regression environments."""

from collections import OrderedDict

import sklearn.datasets

from ..data_types import FeatureType, TargetType, DataSourceType


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
        "target_preprocessor": None,
        "source": DataSourceType.SKLEARN
    }),
    (sklearn.datasets.load_diabetes, {
        "dataset_name": "diabetes",
        "target_type": TargetType.REGRESSION,
        "feature_types": [FeatureType.CONTINUOUS for _ in range(11)],
        "feature_indices": range(11),
        "target_preprocessor": None,
        "source": DataSourceType.SKLEARN
    }),
    (sklearn.datasets.load_linnerud, {
        "dataset_name": "linnerud",
        "target_type": TargetType.REGRESSION,
        "feature_types": [FeatureType.CONTINUOUS for _ in range(3)],
        "feature_indices": range(3),
        "target_preprocessor": None,
        "source": DataSourceType.SKLEARN
    }),
])
