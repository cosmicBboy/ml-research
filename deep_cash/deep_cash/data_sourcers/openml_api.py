"""Source datasets from OpenML.

https://www.openml.org/search?type=data
"""

import numpy as np

import openml

from ..data_types import FeatureType, TargetType, OpenMLTaskType, \
    DataSourceType
from ..errors import TargetNotCompatible


N_CLASSIFICATION_ENVS = 10
N_REGRESSION_ENVS = 10

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
        "target_type": TargetType.MULTICLASS,
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


def list_clf_datasets(n_results=None, exclude_missing=True):
    """Get classification datasets.

    :param int n_results: number of dataset metadata to list
    :param bool exclude_missing: whether or not to exclude datasets with any
        missing values (default=True).
    :returns: OpenML datasets for classification tasks.
    :rtype: list[openml.OpenMLDataset]
    """
    # TODO: figure out how to specify <n and >n filter operators based on
    # the docs: https://www.openml.org/api_docs#!/data/get_data_list_filters
    kwargs = {"number_classes": "%d..%d" % CLASS_RANGE}
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
        name, target_type, data, target, feature_types, feature_indices,
        target_preprocessor):
    return {
        "dataset_name": name,
        "target_type": target_type,
        "data": data,
        "target": target,
        "feature_types": feature_types,
        "feature_indices": feature_indices,
        "target_preprocessor": target_preprocessor,
        "source": DataSourceType.OPEN_ML
    }


def _parse_openml_dataset(
        openml_dataset, target_column, target_type, target_preprocessor=None,
        include_features_with_na=False):
    feature_indices = []
    feature_types = []
    target_index = None
    ignore_atts = openml_dataset.ignore_attributes
    # include row id ensures that indices are correctly aligned with the data
    row_id_att = openml_dataset.row_id_attribute
    data = openml_dataset.get_data(include_row_id=True)
    for key, feature in openml_dataset.features.items():
        if (ignore_atts and feature.name in ignore_atts) or \
                (row_id_att and feature.name == row_id_att):
            continue

        if feature.name == target_column:
            target_index = key
        if feature.number_missing_values != 0:
            if include_features_with_na:
                feature_indices.append(key)
                feature_types.append(_map_feature(feature))
            else:
                continue
        else:
            feature_indices.append(key)
            feature_types.append(_map_feature(feature))
    if target_index is None:
        raise ValueError("target_index must be defined.")
    return _open_ml_dataset(
        name=openml_dataset.name,
        target_type=target_type,
        data=data[:, feature_indices],
        target=data[:, target_index],
        feature_types=feature_types,
        feature_indices=feature_indices,
        # this should only be specified for multilabel problems
        target_preprocessor=target_preprocessor,
    )


def parse_clf_dataset(dataset, dataset_metadata=None):
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
            target_type=custom_config["target_type"],
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
            target_type = TargetType.BINARY
        elif n_classes > 2:
            target_type = TargetType.MULTICLASS
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
                if feature.data_type != "nominal":
                    raise TargetNotCompatible(
                        'dataset "%s" has a continuous target, skipping...' %
                        dataset.name)
                target_index = key
            else:
                feature_indices.append(key)
                feature_types.append(_map_feature(feature))

        if target_index is None:
            raise ValueError("target_index must be defined.")
        return _open_ml_dataset(
            name=dataset.name,
            target_type=target_type,
            data=data[:, feature_indices],
            target=data[:, target_index],
            feature_types=feature_types,
            feature_indices=feature_indices,
            target_preprocessor=None,  # this should only be specified for
                                       # multilabel problems
        )
    else:
        raise ValueError("Cannot parse dataset: %s" % dataset.name)


def classification_envs(n=N_CLASSIFICATION_ENVS):
    clf_dataset_metadata = list_clf_datasets(n)
    clf_dataset_ids = [did for did in clf_dataset_metadata]
    datasets = get_datasets(ids=clf_dataset_ids + list(CUSTOM_DATASET_IDS))
    out = []
    for ds in datasets:
        try:
            out.append(parse_clf_dataset(ds, clf_dataset_metadata))
        except TargetNotCompatible as e:
            # TODO: log this
            print("DATA ENVIRONMENT ERROR: %s" % e)
            continue
    return out


def regression_envs(n=N_REGRESSION_ENVS):
    reg_dataset_metadata = openml.tasks.list_tasks(
        task_type_id=OpenMLTaskType.SUPERVISED_REGRESSION.value,
        size=n)
    dataset_ids = [v["source_data"] for v in reg_dataset_metadata.values()]
    reg_datasets = openml.datasets.get_datasets(dataset_ids)
    target_features = [
        v["target_feature"] for v in reg_dataset_metadata.values()]
    out = []
    for openml_dataset, target_column in zip(reg_datasets, target_features):
        out.append(_parse_openml_dataset(
            openml_dataset, target_column, TargetType.REGRESSION))
    return out


def envs():
    return classification_envs() + regression_envs()
