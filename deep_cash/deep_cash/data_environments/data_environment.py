"""Module to handle logic for data environments."""

import numpy as np
import pandas as pd

from collections import namedtuple

from ..data_types import FeatureType


DataEnvSample = namedtuple(
    "DataEnvSample", [
        "X_train",
        "y_train",
        "X_validation",
        "y_validation",
    ])


def handle_missing_categorical(x):
    """Handle missing data in categorical variables.

    The current implementation will treat categorical missing values as a
    valid "None" category.
    """
    vmap = {}

    try:
        x = x.astype(float)
    except ValueError as e:
        if not e.args[0].startswith("could not convert string to float"):
            raise e
        pass

    x_vals = x[~pd.isnull(x)]
    for i, v in enumerate(np.unique(x_vals)):
        vmap[v] = i

    # null values assume the last category
    if pd.isnull(x).any():
        vmap[None] = len(x_vals)

    # convert values to normalized categories
    x_cat = np.zeros_like(x)
    for i in range(x.shape[0]):
        x_cat[i] = vmap.get(
            None if (isinstance(x[i], float) and np.isnan(x[i])) else x[i])
    return x_cat


def create_simple_date_features(x):
    """Create of simple date features.

    Creates five` numerical features from a datetime:
    - year
    - month-of-year
    - day-of-year
    - day-of-month
    - day-of-week
    """
    x_dates = pd.to_datetime(x)
    return np.hstack([
        x_dates.year.values.reshape(-1, 1),
        x_dates.month.values.reshape(-1, 1),
        x_dates.dayofyear.values.reshape(-1, 1),
        x_dates.day.values.reshape(-1, 1),  # month
        x_dates.dayofweek.values.reshape(-1, 1),
    ]).astype(float)


def preprocess_features(features, feature_types):
    """Prepare data environment for use by task environment.

    :param numpy.array features: rows are training instances and columns are
        features.
    :param list[FeatureType] feature_types: specifies feature types per column.
    """
    clean_features = []
    for i, ftype in enumerate(feature_types):
        x_i = features[:, i]
        if ftype == FeatureType.CATEGORICAL:
            clean_x = handle_missing_categorical(x_i)
        elif ftype == FeatureType.DATE:
            clean_x = create_simple_date_features(x_i)
        else:
            clean_x = x_i.astype(float)

        if len(clean_x.shape) == 1:
            clean_x = clean_x.reshape(-1, 1)
        clean_features.append(clean_x)
    clean_features = np.hstack(clean_features)
    return clean_features


class DataEnvironment(object):

    def __init__(
            self, name, source, target_type, feature_types, feature_indices,
            fetch_training_data, fetch_test_data=None,
            target_preprocessor=None, scorer=None):
        self.name = name
        self.source = source
        self.target_type = target_type
        self.feature_types = feature_types
        self.feature_indices = feature_indices
        self.fetch_training_data = fetch_training_data
        self.fetch_test_data = fetch_test_data
        self.target_preprocessor = target_preprocessor
        self._scorer = scorer
        self._features = None
        self._target = None

    @property
    def training_data(self):
        if self._features is None and self._target is None:
            _features, _target = self.fetch_training_data()
            if self.target_preprocessor is not None:
                # NOTE: this feature isn't really executed currently since none
                # of the classification environments have a specified
                # target_preprocessor. This capability would be used in the
                # case of multi-output tasks.
                _target = self.target_preprocessor().fit_transform(_target)
            _features = preprocess_features(_features, self.feature_types)
            self._features = _features
            self._target = _target
        return self._features, self._target

    @property
    def test_data(self):
        features = self._fetch_test_data()
        return features

    @property
    def scorer(self):
        return self._scorer

    def sample(self, n=None):
        features, target = self.training_data
        if n and features.shape[0] > n:
            # select subset of training data (mainly for large datasets)
            idx = np.random.choice(range(features.shape[0]), n)
            features, target = features[idx], target[idx]

        data_idx = np.array(range(features.shape[0]))
        # bootstrap sampling
        # TODO: support other kinds of cross validation
        train_index = np.random.choice(data_idx, len(data_idx), replace=True)
        validation_index = np.setdiff1d(data_idx, train_index)
        return DataEnvSample(
            features[train_index],
            target[train_index],
            features[validation_index],
            target[validation_index],
        )
