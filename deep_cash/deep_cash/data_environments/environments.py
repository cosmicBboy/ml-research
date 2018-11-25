"""Data environments module."""

from collections import ChainMap

from ..data_types import DataSourceType
from . import openml_api, kaggle_api, sklearn_classification, \
    sklearn_regression


def sklearn_data_envs(dataset_names):
    """Create list of sklearn data environments."""
    envs = ChainMap(sklearn_classification.envs(), sklearn_regression.envs())
    if dataset_names is None:
        return [envs[d] for d in envs]
    return [envs[d] for d in dataset_names]


ENV_SOURCES = {
    # TODO: these wrappers should take a list of dataset names as arguments
    # and return only those dataset envs. Make this efficient by only creating
    # the data envs specified.
    DataSourceType.SKLEARN: sklearn_data_envs,
    DataSourceType.OPEN_ML: openml_api.envs,
    DataSourceType.KAGGLE: kaggle_api.envs,
}


def envs(sources=None, dataset_names=None, target_types=None, n_samples=None):
    """Get classification environments."""
    # TODO: need to set aside test set for each data env. This should be
    # standardized (set a random seed) so that results are reproducible.
    _envs = []
    if sources is None:
        sources = list(ENV_SOURCES.keys())
    for env_source in sources:
        _envs.extend(ENV_SOURCES[env_source](dataset_names))
    if target_types:
        _envs = [e for e in _envs if e.target_type in target_types]
    # NOTE: eventually the MLF pipeline should be able to transform
    # numerical and categorical features selectively, as shown in this example
    # http://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html
    return _envs
