"""Source datasets from OpenML.

https://www.openml.org/search?type=data
"""

import numpy as np

import openml

from ..data_types import FeatureType, TargetType


N_CLASSIFICATION_ENVS = 10

# specifies range of classification tasks from 2-100 classes
CLASS_RANGE = (2, 100)


# NOTE: the key is the OpenML dataset_id
# these are special case datasets that need to be slightly processed in order
# to be usable by deep_cash. The single special case for now is a dataset
# called "anneal", which was NaN values for its categorical data, however,
# these values are actually "not applicable", i.e. they are legitimate discrete
# values in the categorical distribution.
CUSTOM_DATA_CONFIG = {
    2: {
        "dataset_name": "anneal",
        "task_type": TargetType.MULTICLASS,
        "target_index": 38,
        "target_name": "class",
        "target_preprocessor": None,
        "fill_na_discrete": True,
    }
}

CUSTOM_DATASET_IDS = frozenset(CUSTOM_DATA_CONFIG.keys())

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


def list_classification_datasets(n_results=None, exclude_missing=True):
    """Get classification datasets.

    :param int n_results: number of dataset metadata to list
    :param bool exclude_missing: whether or not to exclude datasets with
        with any missing values (default=True).
    :returns: OpenML datasets for classification tasks.
    :rtype: list[openml.OpenMLDataset]
    """
    # TODO: figure out how to specify <n and >n filter operators based on
    # the docs: https://www.openml.org/api_docs#!/data/get_data_list_filters
    kwargs = {
        "number_classes": "%d..%d" % CLASS_RANGE,
    }
    if n_results:
        kwargs.update({"size": n_results})
    if exclude_missing:
        kwargs.update({"number_missing_values": 0})
    return openml.datasets.list_datasets(**kwargs)


def get_datasets(ids=None):
    if ids is None:
        ids = CUSTOM_DATASET_IDS
    return openml.datasets.get_datasets(ids)


def _open_ml_dataset(
        name, task_type, data, target, feature_types, feature_indices,
        target_preprocessor):
    return {
        "dataset_name": name,
        "task_type": task_type,
        "data": data,
        "target": target,
        "feature_types": feature_types,
        "feature_indices": feature_indices,
        "target_preprocessor": target_preprocessor,
    }


def parse_dataset(dataset, dataset_metadata=None):
    """Parses OpenML dataset to follow the sklearn general dataset API.

    For more information, see:
    http://scikit-learn.org/stable/datasets/index.html#general-dataset-api
    """
    custom_config = CUSTOM_DATA_CONFIG.get(dataset.dataset_id, None)
    dataset_metadata = {} if dataset_metadata is None else dataset_metadata
    metadata = dataset_metadata.get(dataset.dataset_id, None)
    data = dataset.get_data()
    if custom_config:
        dataset.features.pop(custom_config["target_index"])
        feature_types = [_map_feature(f) for f in dataset.features.values()]
        feature_indices = list(dataset.features.keys())

        if custom_config["fill_na_discrete"]:
            for f, i in zip(feature_types, feature_indices):
                if f == FeatureType.CATEGORICAL:
                    data[:, i] = _reassign_categorical_data(data[:, i])
        return _open_ml_dataset(
            name=custom_config["dataset_name"],
            task_type=custom_config["task_type"],
            data=data[:, feature_indices],
            target=data[:, custom_config["target_index"]],
            feature_types=feature_types,
            feature_indices=feature_indices,
            target_preprocessor=custom_config["target_preprocessor"],
        )
    elif metadata:
        target_column = dataset.default_target_attribute
        feature_indices = []
        feature_types = []
        target_index = None
        ignore_atts = dataset.ignore_attributes

        n_classes = metadata["NumberOfClasses"]
        if n_classes == 2:
            task_type = TargetType.BINARY
        elif n_classes > 2:
            task_type = TargetType.MULTICLASS
        else:
            raise ValueError(
                "number of classes must be >= 2, found %d" % n_classes)

        for key, feature in dataset.features.items():
            if ignore_atts and feature.name in ignore_atts:
                continue
            if feature.number_missing_values != 0:
                raise ValueError(
                    "deep_cash currently does not support data with missing "
                    "values.")

            if feature.name == target_column:
                target_index = key
            else:
                feature_indices.append(key)
                feature_types.append(_map_feature(feature))

        if target_index is None:
            raise ValueError("target_index must be defined.")
        return _open_ml_dataset(
            name=dataset.name,
            task_type=task_type,
            data=data[:, feature_indices],
            target=data[:, target_index],
            feature_types=feature_types,
            feature_indices=feature_indices,
            target_preprocessor=None,  # this should only be specified for
                                       # multilabel problems
        )
    else:
        raise ValueError("Cannot parse dataset: %s" % dataset.name)


def classification_envs():
    clf_dataset_metadata = list_classification_datasets(N_CLASSIFICATION_ENVS)
    clf_dataset_ids = [did for did in clf_dataset_metadata]
    custom_dataset_ids = [
        i for i, c in CUSTOM_DATA_CONFIG.items()
        if c["task_type"] in [TargetType.BINARY, TargetType.MULTICLASS]]
    datasets = get_datasets(ids=clf_dataset_ids + list(CUSTOM_DATASET_IDS))
    return [parse_dataset(d, clf_dataset_metadata) for d in datasets]
