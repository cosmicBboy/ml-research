"""Data environments module."""

import itertools

from . import classification_environments, regression_environments
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
    "sklearn": sklearn_data_envs,
    "openml": openml_api.classification_envs,
}


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
    return _envs
