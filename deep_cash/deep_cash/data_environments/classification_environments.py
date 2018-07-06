"""Module for sampling from a distribution of classification environments."""

from collections import OrderedDict

import sklearn.datasets
import sklearn.preprocessing

from ..data_types import FeatureType, TargetType
from ..data_sourcers import openml_api


SKLEARN_DATA_ENV_CONFIG = OrderedDict([
    (sklearn.datasets.load_iris, {
        "dataset_name": "iris",
        "task_type": TargetType.MULTICLASS,
        "feature_types": [FeatureType.CONTINUOUS for _ in range(4)],
        "feature_indices": range(4),
        "target_preprocessor": None}),
    (sklearn.datasets.load_digits, {
        "dataset_name": "digits",
        "task_type": TargetType.MULTICLASS,
        "feature_types": [FeatureType.CONTINUOUS for _ in range(64)],
        "feature_indices": range(64),
        "target_preprocessor": None}),
    (sklearn.datasets.load_wine, {
        "dataset_name": "wine",
        "task_type": TargetType.MULTICLASS,
        "feature_types": [FeatureType.CONTINUOUS for _ in range(13)],
        "feature_indices": range(13),
        "target_preprocessor": None}),
    (sklearn.datasets.load_breast_cancer, {
        "dataset_name": "breast_cancer",
        "task_type": TargetType.BINARY,
        "feature_types": [FeatureType.CONTINUOUS for _ in range(30)],
        "feature_indices": range(30),
        "target_preprocessor": None}),
])


def sklearn_classification_envs():
    out = []
    for env_fn, config in SKLEARN_DATA_ENV_CONFIG.items():
        data = env_fn()
        c = config.copy()
        c.update({
            "data": data["data"],
            "target": data["target"],
        })
        out.append(c)
    return out


def envs(names=None):
    _envs = sklearn_classification_envs() + openml_api.classification_envs()
    if names:
        _envs = [e for e in _envs if e["dataset_name"] in names]
    return _envs


def env_names():
    return [n["dataset_name"] for n in envs()]
