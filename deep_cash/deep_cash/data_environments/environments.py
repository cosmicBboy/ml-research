"""Data environments module."""

import itertools
import numpy as np
import pandas as pd

from . import classification_environments, regression_environments
from ..data_types import DataSourceType, FeatureType
from ..data_sourcers import openml_api, kaggle_api


def sklearn_data_envs():
    """Create list of sklearn data environments."""
    out = []
    for env_fn, config in itertools.chain(
            classification_environments.SKLEARN_DATA_ENV_CONFIG.items(),
            regression_environments.SKLEARN_DATA_ENV_CONFIG.items()):
        data = env_fn()
        c = config.copy()
        c.update({
            "data": data["data"],
            "target": data["target"],
        })
        out.append(c)
    return out


ENV_SOURCES = {
    DataSourceType.SKLEARN: sklearn_data_envs,
    DataSourceType.OPEN_ML: openml_api.envs,
    DataSourceType.KAGGLE: kaggle_api.envs,
}


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


def preprocess_data_env(data_env):
    """Prepare data environment for use by task environment."""
    clean_data = []
    for i, ftype in enumerate(data_env["feature_types"]):
        x_i = data_env["data"][:, i]
        if ftype == FeatureType.CATEGORICAL:
            clean_x = handle_missing_categorical(x_i)
        elif ftype == FeatureType.DATE:
            clean_x = create_simple_date_features(x_i)
        else:
            clean_x = x_i.astype(float)

        if len(clean_x.shape) == 1:
            clean_x = clean_x.reshape(-1, 1)
        clean_data.append(clean_x)
    clean_data = np.hstack(clean_data)
    # TODO: this should be a different property in the dictionary.
    data_env["data"] = clean_data
    return data_env


def envs(sources=None, names=None, target_types=None, n_samples=None):
    """Get classification environments."""
    _envs = []
    if sources is None:
        sources = list(ENV_SOURCES.keys())
    for env_source in sources:
        _envs.extend(ENV_SOURCES[env_source]())
    if names:
        _envs = [e for e in _envs if e["dataset_name"] in names]
    if target_types:
        _envs = [e for e in _envs if e["target_type"] in target_types]
    # NOTE: eventually the MLF pipeline should be able to transform
    # numerical and categorical features selectively, as shown in this example
    # http://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html
    return [preprocess_data_env(e) for e in _envs]
