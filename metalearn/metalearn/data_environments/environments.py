"""Data environments module."""

from collections import ChainMap
from functools import partial

from ..data_types import DataSourceType
from . import openml_api, kaggle_api, sklearn_classification, \
    sklearn_regression


def get_envs_from_source(
        env_fns, dataset_names, n=None, test_size=None, random_state=None,
        verbose=None):
    envs = ChainMap(*[
        env_fn(n=n, test_size=test_size, random_state=random_state,
               verbose=verbose)
        for env_fn in env_fns])
    if dataset_names is None:
        return [envs[d] for d in envs]
    return [envs[d] for d in dataset_names if d in envs]


ENV_SOURCES = {
    # TODO: these wrappers should take a list of dataset names as arguments
    # and return only those dataset envs. Make this efficient by only creating
    # the data envs specified.
    DataSourceType.SKLEARN: partial(
        get_envs_from_source,
        [sklearn_classification.envs,
         sklearn_regression.envs]),
    DataSourceType.OPEN_ML: partial(
        get_envs_from_source,
        [openml_api.classification_envs,
         openml_api.regression_envs]),
    DataSourceType.KAGGLE: partial(
        get_envs_from_source,
        [kaggle_api.classification_envs,
         kaggle_api.regression_envs]),
    DataSourceType.AUTOSKLEARN_BENCHMARK: partial(
        get_envs_from_source,
        [openml_api.autosklearn_paper_classification_envs])
}


def envs(dataset_names=None, sources=None, target_types=None,
         test_set_config=None, env_source_map=ENV_SOURCES):
    """Get environments.

    :param list[DataSourceType] sources: only get data envs from these sources.
    :param list[str] dataset_names: only include these datasets in data dist.
    :param list[TargetType] target_types: only include these target types
        in data dist.
    :param dict[DataSourceType -> dict] test_set_config: a dictionary where
        keys DataSourceTypes and values are dictionaries of the form:
        {"test_size": float, "random_state": int}. This is used to set the
        task environment test set sizes.
    """
    # TODO: need to set aside test set for each data env. This should be
    # standardized (set a random seed) so that results are reproducible.
    test_set_config = {} if test_set_config is None else test_set_config
    _envs = []
    if sources is None:
        sources = list(env_source_map.keys())
    for env_source in sources:
        _envs.extend(env_source_map[env_source](
            dataset_names, **test_set_config.get(env_source, {})))
    if target_types:
        _envs = [e for e in _envs if e.target_type in target_types]
    return _envs
