"""Data environments module."""

import itertools
import numpy as np

from . import classification_environments, regression_environments
from ..data_types import DataSourceType, FeatureType
from ..data_sourcers import openml_api


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
}


def handle_missing_categorical(x):
    """Handle missing data in categorical variables.

    The current implementation will treat categorical missing values as a
    valid "None" category.
    """
    vmap = {}
    for i, v in enumerate(np.unique(x)):
        if None in vmap:
            continue
        elif np.isnan(v):
            vmap[None] = i
        else:
            vmap[v] = i
    # convert values to normalized categories
    x_cat = np.zeros_like(x)
    for i in range(x.shape[0]):
        x_cat[i] = vmap.get(None if np.isnan(x[i]) else x[i])
    return x_cat


def preprocess_data_env(data_env):
    """Prepare data environment for use by task environment."""
    for i, ftype in enumerate(data_env["feature_types"]):
        if ftype == FeatureType.CATEGORICAL:
            data_env["data"][:, i] = handle_missing_categorical(
                data_env["data"][:, i])
    return data_env


def envs(sources=None, names=None, target_types=None):
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
    # TODO: preprocess all dataset environments here:
    # - encode missing values in categorical features as their own category.
    #   this is the simplest treatment, in the future we'll want the MLF to
    #   handle imputation of categorical variables.
    # - missing values in numeric features should be left alone, since the
    #   the imputer will handle these cases.

    # NOTE: eventually the MLF pipeline should be able to transform
    # numerical and categorical features selectively, as shown in this example
    # http://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html
    return [preprocess_data_env(e) for e in _envs]
