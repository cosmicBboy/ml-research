"""Kaggle Base Class."""

import pandas as pd
import subprocess

from pathlib import Path

from ..data_types import DataSourceType, FeatureType


CACHE_DIR = "~/.kaggle/cache"
COMPETITION_URL = "https://www.kaggle.com/c"


class KaggleCompetition(object):
    """API for accessing kaggle competition data."""

    def __init__(
            self,
            competition_id,
            features,
            target,
            training_data_fname,
            test_data_fname,
            file_format="csv",
            cache=CACHE_DIR,
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

    def download_cache(self):
        """Downlaod cache to local drive."""
        if not self._dataset_filepath.exists():
            self._dataset_filepath.mkdir()
        subprocess.call(self._download_api_cmd(), cwd=self._dataset_filepath)

    @property
    def url(self):
        """Return url to kaggle dataset."""
        return "%s/%s" % (COMPETITION_URL, self._competition_id)

    @property
    def _dataset_filepath(self):
        return self._cache / self._competition_id

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

    def get_training_data(self):
        """Return a dataframe containing the training set.

        :returns: pandas.DataFrame of training data. Includes features and
            target.
        """
        self.download_cache()
        dataset = pd.read_csv(
            self._dataset_filepath / self._training_data_fname,
            parse_dates=self._datetime_features)
        feature_data = dataset[self._feature_names].values
        target_data = dataset[[i for i in self._target]].values

        if target_data.shape[1] == 1:
            target_data = target_data.ravel()
        return feature_data, target_data

    def get_test_data(self, custom_preprocessor=None):
        """Return a datafrae containing test set.

        Note that this dataset does not contain targets because it is used for
        evaluation on the kaggle platform. This method should only be used
        when evaluating the final test set performance of a model after
        hyperparameter tuning.

        :returns: pandas.DataFrame of test data. Only includes features.
        """
        pass

    def create_data_env(self):
        """Create dictionary with keys matching _open_ml_dataset output."""
        feature_data, target_data = self.get_training_data()
        return {
            "dataset_name": self._competition_id.replace("-", "_"),
            "target_type": self._target_type,
            "data": feature_data,
            "target": target_data,
            "feature_types": self._feature_types,
            "feature_indices": [i for i in range(len(self._feature_types))],
            "target_preprocessor": None,
            "source": DataSourceType.KAGGLE,
        }
