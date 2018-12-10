"""Source datasets from OpenML.

https://www.openml.org/search?type=data
"""

import logging

from collections import OrderedDict
from functools import partial

import openml

from openml.exceptions import OpenMLServerException

from .data_environment import DataEnvironment
from ..data_types import FeatureType, TargetType, OpenMLTaskType, \
    DataSourceType


logger = logging.getLogger(__name__)


N_CLASSIFICATION_ENVS = 20
N_REGRESSION_ENVS = 20


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


def get_datasets(dataset_ids):
    datasets = []
    for did in dataset_ids:
        try:
            datasets.append(openml.datasets.get_dataset(did))
        except OpenMLServerException as e:
            logger.info(
                "OpenML server exception on dataset_id %s, %s" %
                (did, e))
    return datasets


def openml_to_data_env(
        openml_dataset, target_column, target_type, test_size, random_state,
        target_preprocessor=None, include_features_with_na=False):
    feature_indices = []
    feature_types = []
    target_index = None
    ignore_atts = openml_dataset.ignore_attributes
    row_id_att = openml_dataset.row_id_attribute
    for key, feature in openml_dataset.features.items():
        if (ignore_atts and feature.name in ignore_atts) or \
                (row_id_att and feature.name == row_id_att):
            print("ignoring attribute %s in dataset %s" %
                  (ignore_atts, openml_dataset.name))
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
        print("target column %s not found for dataset %s, skipping." %
              (target_column, openml_dataset.name))
        return None
    if len(feature_indices) == 0:
        print("no features found for dataset %s, skipping." %
              openml_dataset.name)
        return None

    def _fetch_training_data():
        data = openml_dataset.get_data(include_row_id=True)
        return data[:, feature_indices], data[:, target_index]

    return DataEnvironment(
        name="openml.%s" % openml_dataset.name.lower(),
        source=DataSourceType.OPEN_ML,
        target_type=target_type,
        feature_types=feature_types,
        feature_indices=feature_indices,
        # include row id ensures that indices are correctly aligned
        fetch_training_data=_fetch_training_data,
        fetch_test_data=None,
        test_size=test_size,
        random_state=random_state,
        target_preprocessor=target_preprocessor,
        scorer=None,
    )


def classification_envs(
        n=N_CLASSIFICATION_ENVS, test_size=None, random_state=None):
    clf_dataset_metadata = openml.tasks.list_tasks(
        task_type_id=OpenMLTaskType.SUPERVISED_CLASSIFICATION.value, size=n)
    dataset_ids = [v["source_data"] for v in clf_dataset_metadata.values()]
    clf_datasets = openml.datasets.get_datasets(dataset_ids)
    target_features, target_types = zip(*[
        (v["target_feature"],
         TargetType.MULTICLASS if v["NumberOfClasses"] > 2
            else TargetType.BINARY)
        for v in clf_dataset_metadata.values()])
    envs = []
    for ds, tc, tt in zip(clf_datasets, target_features, target_types):
        parsed_ds = openml_to_data_env(
            ds, tc, tt, test_size, random_state, include_features_with_na=True)
        if parsed_ds:
            envs.append(parsed_ds)
    return OrderedDict([(d.name, d) for d in envs])


def regression_envs(
        n=N_REGRESSION_ENVS, test_size=None, random_state=None):
    reg_dataset_metadata = openml.tasks.list_tasks(
        task_type_id=OpenMLTaskType.SUPERVISED_REGRESSION.value, size=n)
    dataset_ids = [v["source_data"] for v in reg_dataset_metadata.values()]
    reg_datasets = get_datasets(dataset_ids)
    target_features = [
        v["target_feature"] for v in reg_dataset_metadata.values()]
    envs = []
    for openml_dataset, target_column in zip(reg_datasets, target_features):
        parsed_ds = openml_to_data_env(
            openml_dataset, target_column, TargetType.REGRESSION,
            test_size, random_state, include_features_with_na=True)
        if parsed_ds:
            envs.append(parsed_ds)
    return OrderedDict([(d.name, d) for d in envs])
