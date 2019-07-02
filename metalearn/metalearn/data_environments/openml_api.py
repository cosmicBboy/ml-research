"""Source datasets from OpenML.

https://www.openml.org/search?type=data
"""

import logging

from collections import OrderedDict
from functools import partial

import numpy as np
import openml

from openml.exceptions import OpenMLServerException

from .data_environment import DataEnvironment
from . import autosklearn_clf_task_ids
from ..data_types import FeatureType, TargetType, OpenMLTaskType, \
    DataSourceType


logger = logging.getLogger(__name__)


N_CLASSIFICATION_ENVS = 20
N_REGRESSION_ENVS = 20
SPARSE_DATA_FORMAT = "Sparse_ARFF"


# Map OpenML feature data types to metalearn feature data types
FEATURE_TYPE_MAP = {
    "nominal": FeatureType.CATEGORICAL,
    "numeric": FeatureType.CONTINUOUS,
}
DATASOURCE_TYPES = [
    DataSourceType.OPEN_ML, DataSourceType.AUTOSKLEARN_BENCHMARK]


def _map_feature(feature):
    feature_dtype = FEATURE_TYPE_MAP.get(feature.data_type, None)
    if feature_dtype is None:
        raise ValueError(
            "feature dtype %s not recognized. valid values are %s" % (
                FEATURE_TYPE_MAP.keys()))
    return feature_dtype


def _normalize_feature_indices(feature_indices):
    min_index = min(feature_indices)
    return [i - min_index for i in feature_indices]


def openml_to_data_env(
        openml_dataset, target_column, target_type,
        data_source_type, test_size, random_state, target_preprocessor=None,
        include_features_with_na=False):
    print("fetching openml dataset: %s" % openml_dataset.name)
    if data_source_type not in DATASOURCE_TYPES:
        raise ValueError(
            "expected one of %s, found : %s" % (
                DATASOURCE_TYPES, data_source_type))
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
            continue
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
        # see https://openml.github.io/openml-python/develop/generated/openml.OpenMLDataset.html#openml.OpenMLDataset  # noqa E53
        # for more details
        data, *_ = openml_dataset.get_data(
            include_row_id=True, include_ignore_attributes=True,
            dataset_format="array")
        if openml_dataset.format == SPARSE_DATA_FORMAT:
            # TODO: currently the controller can't handle sparse matrices.
            data = data.todense()

        # remove rows with null targets. Only case we'll need that is if
        # metalearn supports semi-supervised learning.
        X = data[:, feature_indices]
        y = data[:, target_index].ravel()
        y_not_nan = ~np.isnan(y)
        return X[y_not_nan, :], y[y_not_nan]

    # normalize feature indices so that index is aligned with
    # _fetch_training_data
    min_index = min(feature_indices)
    normalized_feature_indices = [i - min_index for i in feature_indices]

    return DataEnvironment(
        name="openml.%s" % openml_dataset.name.lower(),
        source=data_source_type,
        target_type=target_type,
        feature_types=feature_types,
        feature_indices=normalized_feature_indices,
        # include row id ensures that indices are correctly aligned
        fetch_training_data=_fetch_training_data,
        fetch_test_data=None,
        test_size=test_size,
        random_state=random_state,
        target_preprocessor=target_preprocessor,
        scorer=None,
    )


def _get_target_type(num_classes, task_type):
    if task_type is OpenMLTaskType.SUPERVISED_REGRESSION:
        return TargetType.REGRESSION
    elif task_type is OpenMLTaskType.SUPERVISED_CLASSIFICATION:
        if num_classes > 2:
            return TargetType.MULTICLASS
        else:
            return TargetType.BINARY
    else:
        raise ValueError(
            "task_type %s is currently not recognized by metalearn." %
            task_type)


def _get_dataset_metadata(openml_task_metadata, task_type):
    target_columns, target_types = zip(*[
        (v["target_feature"],
         _get_target_type(v.get("NumberOfClasses", 0), task_type))
        for v in openml_task_metadata.values()])
    dataset_ids = [v["source_data"] for v in openml_task_metadata.values()]
    datasets, target_columns_out, target_types_out = [], [], []
    for id, tcols, ttypes in zip(dataset_ids, target_columns, target_types):
        try:
            datasets.append(openml.datasets.get_dataset(id))
            target_columns_out.append(tcols)
            target_types_out.append(ttypes)
        except OpenMLServerException as e:
            print("error when fetching openml metadata: %s, skipping dataset"
                  % e)
    return datasets, target_columns_out, target_types_out


def _create_envs(
        data_source_type, datasets, target_columns, target_types, test_size,
        random_state):
    _openml_to_data_env = partial(
        openml_to_data_env,
        data_source_type=data_source_type,
        test_size=test_size,
        random_state=random_state,
        include_features_with_na=True)
    envs = [_openml_to_data_env(*args) for args in zip(
                datasets, target_columns, target_types)]
    return OrderedDict([(d.name, d) for d in envs if d])


def classification_envs(
        n=N_CLASSIFICATION_ENVS, test_size=None, random_state=None,
        verbose=False):
    if n is None:
        n = N_CLASSIFICATION_ENVS
    task_type = OpenMLTaskType.SUPERVISED_CLASSIFICATION
    dataset_metadata = _get_dataset_metadata(
        openml.tasks.list_tasks(task_type_id=task_type.value, size=n),
        task_type)
    return _create_envs(
        DataSourceType.OPEN_ML, *dataset_metadata +
        (test_size, random_state))


def regression_envs(
        n=N_REGRESSION_ENVS, test_size=None, random_state=None, verbose=False):
    if n is None:
        n = N_CLASSIFICATION_ENVS
    task_type = OpenMLTaskType.SUPERVISED_REGRESSION
    dataset_metadata = _get_dataset_metadata(
        openml.tasks.list_tasks(task_type_id=task_type.value, size=n),
        task_type)
    return _create_envs(
        DataSourceType.OPEN_ML, *dataset_metadata + (test_size, random_state))


def _task_ids(n):
    if n is None:
        task_ids = autosklearn_clf_task_ids.TASK_IDS
    else:
        task_ids = autosklearn_clf_task_ids.TASK_IDS[:n]
    return task_ids


def _get_task(id, verbose):
    if verbose:
        print("getting task metadata for task id %d" % id)
    try:
        task = openml.tasks.get_task(id)
        if task is None:
            print("skipping dataset %d id. "
                  "openml.tasks.get_task return null" % id)
        return task
    except OpenMLServerException as e:
        print("skipping dataset %d. error: %s" % (id, e))


def autosklearn_paper_classification_envs(
        n=None, test_size=None, random_state=None, verbose=False):
    """From Feurer et al. "Efficient and Robust Automated Machine Learning.

    https://papers.nips.cc/paper/5872-efficient-and-robust-automated-machine-learning  # noqa
    """
    if n is None:
        task_ids = autosklearn_clf_task_ids.TASK_IDS
    else:
        task_ids = autosklearn_clf_task_ids.TASK_IDS[:n]
    openml_tasks = list(filter(
        None, [_get_task(id, verbose) for id in task_ids]))
    openml_datasets = [t.get_dataset() for t in openml_tasks]
    target_columns = (t.target_name for t in openml_tasks)
    target_types = (_get_target_type(
        len(t.class_labels),
        OpenMLTaskType.SUPERVISED_CLASSIFICATION) for t in openml_tasks)
    return _create_envs(
        DataSourceType.AUTOSKLEARN_BENCHMARK, openml_datasets, target_columns,
        target_types, test_size, random_state)
