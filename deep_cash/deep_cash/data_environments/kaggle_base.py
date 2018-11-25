"""Kaggle Base Class."""

import os
import pandas as pd
import subprocess

from pathlib import Path

from .data_environment import DataEnvironment
from ..data_types import DataSourceType, FeatureType


KAGGLE_CACHE_DIR = os.environ.get("KAGGLE_CACHE_DIR", "~/.kaggle/cache")
KAGGLE_COMPETITION_URL = "https://www.kaggle.com/c"


class KaggleCompetition(object):
    """API for accessing kaggle competition data."""

    def __init__(
            self,
            competition_id,
            features,
            target,
            training_data_fname,
            test_data_fname,
            scorer=None,
            file_format="csv",
            cache=KAGGLE_CACHE_DIR,
            custom_preprocessor=None):
        """Initialize Kaggle Competition object.

        :params str competition_id: kaggle identifier for the competition. This
            is a dash-delimited string, e.g. "allstate-claims-severity"
        :params dict[str -> FeatureType] features: a dictionary mapping feature
            column names to FeatureTypes.
        :params dict[str -> TargetType] target: a dictionary mapping target
            column to TargetType.
        :params str training_data_fname: filename of the training set.
        :params str test_data_fname: filename of the test set.
        :params Scorer|None scorer: a `scorer.Scorer` named tuple that
            specifies how to evaluate the performance of the dataset.
        :params str file_format: currently only csv files are supported.
        :params str cache: directory location for caching competition files.
        :param callable|None custom_preprocessor: a function for doing custom
            pre-processing on the training set. This function should take
            as an argument a dataframe (containing the training/test data) and
            return a dataframe with additional columns. This function should
            be specified for competition datasets with more complex structures
            that involve joining the training/test sets with an auxiliary data
            source.
        """
        self._competition_id = competition_id
        self._features = features
        self._target = target
        self._training_data_fname = training_data_fname
        self._test_data_fname = test_data_fname
        self._scorer = scorer
        if file_format != "csv":
            raise ValueError(
                "'%s' is not a valid file format, only '.csv' currently "
                "supported.")
        self._file_format = file_format
        self._cache = Path(cache).expanduser()
        self._custom_preprocessor = custom_preprocessor

        if not self._cache.exists():
            self._cache.mkdir()

        self._feature_names, self._feature_types = map(
            list, zip(*self._features.items()))

        # currently only support one target.
        if len(self._target) > 1:
            raise ValueError(
                "can only specify one target, found %s" % self.target.keys())
        self._target_name = list(self._target.keys())[0]
        self._target_type = self._target[self._target_name]

    def _download_and_cache(self):
        """Download cache to local drive."""
        if not self._dataset_filepath.exists():
            self._dataset_filepath.mkdir()
        if not (self._training_set_filepath.exists() and
                self._test_set_filepath.exists()):
            subprocess.call(
                self._download_api_cmd(), cwd=self._dataset_filepath)

    @property
    def dataset_name(self):
        return "kaggle.%s" % self._competition_id.replace("-", "_")

    @property
    def url(self):
        """Return url to kaggle dataset."""
        return "%s/%s" % (KAGGLE_COMPETITION_URL, self._competition_id)

    @property
    def scorer(self):
        """Return Scorer object."""
        return self._scorer

    @property
    def _dataset_filepath(self):
        return self._cache / self._competition_id

    @property
    def _training_set_filepath(self):
        return self._dataset_filepath / self._training_data_fname

    @property
    def _test_set_filepath(self):
        return self._dataset_filepath / self._test_data_fname

    @property
    def _datetime_features(self):
        return [k for k, v in self._features.items() if v is FeatureType.DATE]

    def _download_api_cmd(self):
        """Download and cache the competition files in cache directory.

        Note: this command line call checks whether dataset already downloaded
        and skips if true.
        """
        return [
            "kaggle",
            "competitions",
            "download",
            self._competition_id,
        ]

    def get_training_data(self, n_samples=None):
        """Return a dataframe containing the training set.

        :returns: tuple of arrays of training data. First element is an array
            of features, second is an array of targets.
        """
        dataset = pd.read_csv(
            self._training_set_filepath, parse_dates=self._datetime_features,
            nrows=n_samples)

        if self._custom_preprocessor:
            dataset = self._custom_preprocessor(dataset)

        feature_data = dataset[self._feature_names].values
        target_data = dataset[[i for i in self._target]].values

        if target_data.shape[1] == 1:
            target_data = target_data.ravel()
        return feature_data, target_data

    def get_test_data(self, n_samples=None):
        """Return a dataframe containing test set.

        Note that this dataset does not contain targets because it is used for
        evaluation on the kaggle platform. This method should only be used
        when evaluating the final test set performance of a model after
        hyperparameter tuning.

        :returns: pandas.DataFrame of test data. Only includes features.
        """
        dataset = pd.read_csv(
            self._test_set_filepath, parse_dates=self._datetime_features,
            nrows=n_samples)

        if self._custom_preprocessor:
            dataset = self._custom_preprocessor(dataset)

        feature_data = dataset[self._feature_names].values
        return feature_data

    def data_env(self, n_samples=None):
        """Create dictionary with keys matching _open_ml_dataset output."""
        self._download_and_cache()
        return DataEnvironment(
            name=self.dataset_name,
            source=DataSourceType.KAGGLE,
            target_type=self._target_type,
            feature_types=self._feature_types,
            feature_indices=[i for i in range(len(self._feature_types))],
            fetch_training_data=self.get_training_data,
            fetch_test_data=self.get_test_data,
            target_preprocessor=None,
            scorer=self.scorer,
        )
