"""Module to handle logic for data environments."""

import numpy as np
import pandas as pd

from collections import namedtuple
from sklearn.model_selection import train_test_split

from ..data_types import FeatureType, TargetType

NULL_DATA_ENV = "<NULL_DATA_ENV>"


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

    TODO: need to let the ML pipeline handle one-hot encoding of categorical
        features so that the state can properly maintained in the test and
        validation sets. Therefore, all this function should do is replace
        the NA with a special "<NONE_CATEGORY>" token
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
        self.feature_types = feature_types
        self.feature_indices = feature_indices
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
            _features = preprocess_features(_features, self.feature_types)
            self._data_cache[feature_key] = _features
            self._data_cache[target_key] = _target
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
