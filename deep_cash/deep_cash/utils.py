"""Utility functions."""

import logging
import time

import numpy as np
import torch

from torch.autograd import Variable

from .data_environments.classification_environments import env_names


def create_metafeature_spec(env_sources):
    """Create a metafeature spec.

    NOTE: may need to make this a class if it becomes more complex.
    """
    return [
        ("data_env_name", str, env_names(sources=env_sources)),
        ("number_of_examples", int, None),
        ("number_of_features", int, None),
    ]


def get_metafeatures_dim(metafeatures_spec):
    """Get dimensionality of metafeatures."""
    return sum([len(m[2]) if m[1] is str else 1 for m in metafeatures_spec])


def init_logging(log_path="/tmp/deep_cash.log"):
    """Initialize logging at the module level.

    :param str|None log_path: path to write logs. If None, write to stdout
    """
    # remove other logging handlers to re-configure logging.
    for h in logging.root.handlers:
        logging.root.removeHandler(h)
    with open(log_path, "w") as f:
        f.close()
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s: %(levelname)s: %(name)s: %(message)s")
    if log_path is not None:
        logging.info("writing logs to %s" % log_path)


class Timer(object):
    """A light-weight timer class."""

    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.interval = time.clock() - self.start


def load_model(path, model_class, *args, **kwargs):
    """Load a pytorch model."""
    rnn = model_class(*args, **kwargs)
    rnn.load_state_dict(torch.load(path))
    return rnn


def _create_metafeature_tensor(metafeatures, seq, metafeature_spec):
    """Convert a metafeature vector into a tensor.

    :returns Tensor: dim <string_length x 1 x metafeature_dim>, where
        metafeature_dim is a continuous feature.
    """
    m = []
    for i, feature in enumerate(metafeatures):
        fname, ftype, flevels = metafeature_spec[i]
        if ftype is int:
            metafeature_dim = 1
            feature_val = ftype(feature)
            feature_index = 0
        elif ftype is str:
            metafeature_dim = len(flevels)
            feature_val = 1
            feature_index = flevels.index(feature)
        else:
            raise ValueError(
                "metafeature type %s not recognized" % ftype)
        t = torch.zeros(len(seq), 1, metafeature_dim)
        for j, _ in enumerate(seq):
            t[j][0][feature_index] = feature_val
        m.append(t)
    m = torch.cat(m, 2)
    return m


def _create_input_tensor(a_space, seq):
    """Convert sequence to algorithm space.

    Returns tensor of dim <string_length x 1 x n_components>.

    :param AlgorithmSpace a_space: an algorithm space object that specifies
        the space of possible ML framework components and hyperparameters
    :param list[str] seq: sequence of algorithm components to encode into
        integer tensor.
    :returns: integer algorithm input tensor.
    """
    t = torch.zeros(len(seq), 1, a_space.n_components)
    for i, action in enumerate(seq):
        t[i][0][a_space.components.index(action)] = 1
    return t


def _create_training_data_tensors(a_space, metafeatures, seq):
    return (
        Variable(_create_metafeature_tensor(metafeatures, seq)),
        Variable(_create_input_tensor(a_space, seq)))


def _ml_framework_string(ml_framework):
    return " > ".join(s[0] for s in ml_framework.steps)


def _diversity_metric(n_unique, n_total):
    return np.nan if n_total == 0 else \
        0 if n_total == 1 else \
        (n_unique - 1) / (n_total - 1)


def _hyperparameter_string(hyperparameters):
    return "[%d hyperparams]" % len(hyperparameters)


def _exponential_mean(x, x_prev, beta=0.99):
    # TODO: implement bias correction factor to account for underestimate of
    # the moving average with term `exp_mean / (1 - beta^t)` where `t` is a
    # counter of number of time-steps.
    return (x * beta) + (x_prev * (1 - beta))
