"""Source datasets from OpenML.

https://www.openml.org/search?type=data
"""

import numpy as np

import openml

from ..data_types import FeatureType, TargetType


# NOTE: the key is the OpenML dataset_id
DATA_CONFIG = {
    2: {
        "dataset_name": "anneal",
        "task_type": TargetType.MULTICLASS,
        "target_index": 38,
        "target_name": "class",
        "target_preprocessor": None,
        "fill_na_discrete": True,
    }
}

DATASET_IDS = frozenset(DATA_CONFIG.keys())

# Map OpenML feature data types to deep_cash feature data types
FEATURE_TYPE_MAP = {
    "nominal": FeatureType.CATEGORICAL,
    "numeric": FeatureType.CONTINUOUS,
}


def _map_feature(feature):
    feature_dtype = FEATURE_TYPE_MAP.get(feature.data_type, None)
    if feature_dtype is None:
        raise ValueError(
            "feature dtype %s not recognized. valid values are %s" % (
                FEATURE_TYPE_MAP.keys()))
    return feature_dtype


def _reassign_categorical_data(col_vector):
    """Reassigns categorical data.

    Some openml datasets have the following quirk whereNaN values are present
    in the dataset, but in fact these values should be "not applicable", i.e.
    they are a valid discrete values.
    """
    null_values = np.isnan(col_vector)
    if not np.any(null_values):
        return col_vector
    unique = np.unique(col_vector[~null_values])
    fill_na_value = 0 if len(unique) == 0 else np.max(unique) + 1
    # assume that null values that fall under "not applicable" value can take
    # on the (n + 1)th index, where n is the max value of the categorical
    # value index
    col_vector[null_values] = fill_na_value
    return col_vector


def get_datasets(ids=None):
    if ids is None:
        ids = DATASET_IDS
    return openml.datasets.get_datasets(ids)


def parse_dataset(dataset):
    """Parses OpenML dataset to follow the sklearn general dataset API.

    For more information, see:
    http://scikit-learn.org/stable/datasets/index.html#general-dataset-api
    """
    config = DATA_CONFIG[dataset.dataset_id]
    dataset.features.pop(config["target_index"])
    feature_types = [_map_feature(f) for f in dataset.features.values()]
    feature_indices = list(dataset.features.keys())
    data = dataset.get_data()

    if config["fill_na_discrete"]:
        for f, i in zip(feature_types, feature_indices):
            if f == FeatureType.CATEGORICAL:
                data[:, i] = _reassign_categorical_data(data[:, i])
    return {
        "dataset_name": config["dataset_name"],
        "task_type": config["task_type"],
        "data": data[:, feature_indices],
        "target": data[:, config["target_index"]],
        "feature_types": feature_types,
        "feature_indices": feature_indices,
        "target_preprocessor": config["target_preprocessor"],
    }


def classification_envs():
    datasets = get_datasets([
        i for i, c in DATA_CONFIG.items()
        if c["task_type"] in [TargetType.BINARY, TargetType.MULTICLASS]])
    return [parse_dataset(d) for d in datasets]
