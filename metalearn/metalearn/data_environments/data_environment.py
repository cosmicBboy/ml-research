"""Module to handle logic for data environments."""

import logging
import numpy as np
import pandas as pd

from collections import namedtuple, defaultdict
from sklearn.model_selection import train_test_split

from ..data_types import FeatureType, TargetType

from typing import List, Union, Tuple

NULL_DATA_ENV = "<NULL_DATA_ENV>"


logger = logging.getLogger(__name__)


PreprocessedFeatures = namedtuple(
    "PreprocessedFeatures", [
        "X",
        "feature_types",
        "feature_indices",
        "feature_names",
    ], defaults=(None, ))


DataEnvSample = namedtuple(
    "DataEnvSample", [
        "X_train",
        "y_train",
        "X_validation",
        "y_validation",
    ])


def create_simple_date_features(x, name=None):
    """Create of simple date features.

    Creates five` numerical features from a datetime:
    - year
    - month-of-year
    - day-of-year
    - day-of-month
    - day-of-week
    """
    x_dates = pd.to_datetime(x)
    x = [
        x_dates.year.values.reshape(-1, 1),
        x_dates.month.values.reshape(-1, 1),
        x_dates.dayofyear.values.reshape(-1, 1),
        x_dates.day.values.reshape(-1, 1),  # month
        x_dates.dayofweek.values.reshape(-1, 1),
    ]
    if name is not None:
        feature_names = [
            f"{name}_year",
            f"{name}_month",
            f"{name}_day_of_year",
            f"{name}_day",
            f"{name}_day_of_week",
        ]
    else:
        feature_names = None
    return x, feature_names


def _stack_features(
        clean_features: List[np.ndarray],
        feature_types: List[FeatureType],
        feature_names: List[str]=None
        ) -> PreprocessedFeatures:
    """Stack continuous features before categorical features."""
    if feature_names is not None and len(feature_names) != len(feature_types):
        raise ValueError(
            "expected feature_types and feature_names to have the same "
            f"length, found {len(feature_names)} and {len(feature_types)}")
    feature_groups = defaultdict(list)
    feature_name_groups = defaultdict(list)
    for i, (ftype, x) in enumerate(zip(feature_types, clean_features)):
        # all feature types should reduce to categorical and continuous
        feature_groups[ftype].append(x)
        if feature_names is not None:
            feature_name_groups[ftype].append(feature_names[i])

    X, output_feature_types = [], []
    output_feature_names = None if feature_names is None else []
    for ftype in [FeatureType.CONTINUOUS, FeatureType.CATEGORICAL]:
        X.extend(feature_groups[ftype])
        output_feature_types.extend([ftype] * len(feature_groups[ftype]))
        if feature_names is not None:
            output_feature_names.extend(feature_name_groups[ftype])

    return PreprocessedFeatures(
        np.hstack(X),
        output_feature_types,
        list(range(len(output_feature_types))),
        output_feature_names,
    )


def preprocess_features(
        name: str,
        features: np.ndarray,
        feature_types: List[FeatureType],
        feature_names: List[str]=None,
        ) -> PreprocessedFeatures:
    """Prepare data environment for use by task environment.

    :param numpy.array features: rows are training instances and columns are
        features.
    :param list[FeatureType] feature_types: specifies feature types per column.
    """
    clean_features = []
    clean_feature_types = []
    clean_feature_names = None if feature_names is None else []

    for i, ftype in enumerate(feature_types):
        x_i = features[:, i]

        try:
            if np.isnan(x_i).all():
                logger.info(
                    f"feature {features[i]} of type {feature_types[i]} has "
                    f"all null values, dropping from the {name} dataset.")
                continue
        except TypeError:
            pass

        if ftype == FeatureType.CATEGORICAL:
            clean_x = [x_i]
            clean_feature_types.append(FeatureType.CATEGORICAL)
            if feature_names is not None:
                clean_feature_names.append(feature_names[i])

        elif ftype == FeatureType.DATE:
            clean_x, date_features = create_simple_date_features(
                x_i, None if feature_names is None else feature_names[i])
            clean_feature_types.extend(
                [FeatureType.CONTINUOUS for _ in range(len(clean_x))])
            if feature_names is not None:
                clean_feature_names.extend(date_features)

        elif ftype == FeatureType.CONTINUOUS:
            clean_x = [x_i.astype(float)]
            clean_feature_types.append(FeatureType.CONTINUOUS)
            if feature_names is not None:
                clean_feature_names.append(feature_names[i])

        if len(clean_x) == 1 and len(clean_x[0].shape) == 1:
            clean_x = [clean_x[0].reshape(-1, 1)]
        clean_features.extend(clean_x)

    assert len(clean_features) == len(clean_feature_types)
    return _stack_features(
        clean_features, clean_feature_types, clean_feature_names)


def _create_data_partition_fns(fetch_training_data, test_size, random_state):
    X_train, X_test, y_train, y_test = train_test_split(
        *fetch_training_data(), test_size=test_size, random_state=random_state)

    # TODO: figure out a way of doing this lazily without having to actually
    # fetch the training data when data environment is initialized

    def _fetch_training_data():
        return X_train, y_train

    def _fetch_test_data():
        return X_test, y_test

    return _fetch_training_data, _fetch_test_data


class DataEnvironment(object):

    def __init__(
            self, name, source, target_type, feature_types, feature_indices,
            fetch_training_data, fetch_test_data=None, test_size=None,
            random_state=None, target_preprocessor=None, scorer=None):
        """Initialize data environment, used to sample tasks.

        :param str name: name of data env.
        :param DataSourceType source: where data environment was obtained.
        :param TargetType target_type: the target type of data env.
        :param list[FeatureType] feature_types: type of each feature.
        :param list[int] feature_indices: indices in the feature design matrix
            corresponding to the feature type.
        :param callable fetch_training_data: function with no arguments that
            returns an array of X, y where X is the feature matrix and y is
            the target matrix.
        :param callable fetch_test_data: function with no arguments that
            returns an array of X, y where X is the feature matrix and y is
            the target matrix.
        :param float test_size: Number of examples to withhold from the
            training set. This option only works when fetch_test_data is None.
        :param int random_state: random state to use when selecting holdout
            test-set. Only applicable when fetch_test_data is None.
        :param transformer target_preprocessor: an sklearn-compliant target
            preprocessor that has a fit/transform/fit_transform method. This
            is applied to the target

        TODO: add `is_sparse` indicator attribute
        """
        self.name = name
        self.source = source
        self.target_type = target_type
        self.raw_feature_types = feature_types
        self.raw_feature_indices = feature_indices
        if test_size and not 0 < test_size < 1:
            raise ValueError(
                "test_size must be a float between 0 and 1, found: %0.02f"
                % test_size)
        self.test_size = test_size
        self.random_state = random_state
        if fetch_test_data is None and self.test_size is not None:
            fetch_training_data, fetch_test_data = _create_data_partition_fns(
                fetch_training_data, self.test_size, self.random_state)
        self.fetch_training_data = fetch_training_data
        self.fetch_test_data = fetch_test_data
        self.target_preprocessor = target_preprocessor
        self._scorer = scorer
        self._data_cache = {}

        # set by cache_fetch_data
        self.feature_types = None
        self.feature_indices = None

    @property
    def training_data(self):
        return self.cache_fetch_data("train")

    @property
    def test_data(self):
        if self.fetch_test_data is None:
            raise RuntimeError(
                "fetch_test_data attribute is None. Cannot produce a "
                "test set. You need to provide a fetch_test_data function "
                "to the DataEnvironment initializer, or specify the test_size "
                "argument.")
        return self.cache_fetch_data("test")

    @property
    def classes(self):
        if self.target_type is TargetType.MULTICLASS:
            return sorted(list(set(self.training_data[1])))
        return None

    @property
    def X_test(self):
        return self.test_data[0]

    @property
    def y_test(self):
        return self.test_data[1]

    @property
    def scorer(self):
        return self._scorer

    def cache_fetch_data(self, partition):
        feature_key = "%s_features" % partition
        target_key = "%s_target" % partition
        if partition == "train":
            fetch_data_fn = self.fetch_training_data
        elif partition == "test":
            fetch_data_fn = self.fetch_test_data
        else:
            raise ValueError(
                "partition should be 'train' or 'test', found: %s" % partition)
        if feature_key not in self._data_cache and \
                target_key not in self._data_cache:
            _features, _target = fetch_data_fn()
            if self.target_preprocessor is not None:
                _target = self.target_preprocessor().fit_transform(_target)
            _features, feature_types, feature_indices, _ = preprocess_features(
                self.name,
                _features,
                self.raw_feature_types,
                feature_names=None)

            # cache features and target
            self._data_cache[feature_key] = _features
            self._data_cache[target_key] = _target

            # set feature types and indices based on the training partition
            if partition == "train":
                self.feature_types = feature_types
                self.feature_indices = feature_indices
        return self._data_cache[feature_key], self._data_cache[target_key]

    def sample(self, n=None):
        features, target = self.training_data
        if n and features.shape[0] > n:
            # select subset of training data (mainly for large datasets)
            idx = np.random.choice(range(features.shape[0]), n)
            features, target = features[idx], target[idx]

        data_idx = np.array(range(features.shape[0]))
        # bootstrap sampling
        # TODO: support other kinds of data sampling, e.g. cross validation
        train_index = np.random.choice(data_idx, len(data_idx), replace=True)
        validation_index = np.setdiff1d(data_idx, train_index)
        return DataEnvSample(
            features[train_index],
            target[train_index],
            features[validation_index],
            target[validation_index],
        )
