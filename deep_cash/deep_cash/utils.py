"""Utility functions."""

from collections import namedtuple

import torch
from torch.autograd import Variable

# specify type of the metafeature
METAFEATURES = [
    ("number_of_examples", int)
]

PerformanceTracker = namedtuple("PerformanceTracker", [
    "best_candidates", "best_scores", "overall_mean_reward", "overall_a_loss",
    "overall_h_loss", "overall_ml_performance", "running_reward"])


def _create_metafeature_tensor(metafeatures, seq):
    """Convert a metafeature vector into a tensor.

    For now this will be a single category indicating `is_executable`.

    :returns Tensor: dim <string_length x 1 x metafeature_dim>, where
        metafeature_dim is a continuous feature.
    """
    m = []
    for i, feature in enumerate(metafeatures):
        metafeature_type = METAFEATURES[i][1]
        if metafeature_type is int:
            metafeature_dim = 1
            feature = metafeature_type(feature)
        else:
            raise ValueError(
                "metafeature type %s not recognized" % metafeature_type)
        t = torch.zeros(len(seq), 1, metafeature_dim)
        for j, _ in enumerate(seq):
            t[j][0][metafeature_dim - 1] = feature
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


def _create_hyperparameter_tensor(a_space, seq_index):
    """Convert sequence to hyperparameter space.

    Returns tensor of dim <string_length x 1 x n_hyperparameters>.

    :param AlgorithmSpace a_space: an algorithm space object that specifies
        the space of possible ML framework components and hyperparameters
    :param list[int] seq: sequence of hyperparameter values to encode into
        integer tensor.
    :returns: integer hyperparameter input tensor.
    """
    # NOTE: this is probably the way to go instead of encoding strings into
    # integers as _create_input_tensor does.
    t = torch.zeros(len(seq_index), 1, a_space.n_hyperparameters)
    for i, index in enumerate(seq_index):
        t[i][0][index] = 1
    return Variable(t)


def _create_training_data_tensors(a_space, metafeatures, seq):
    return (
        Variable(_create_metafeature_tensor(metafeatures, seq)),
        Variable(_create_input_tensor(a_space, seq)))


def _ml_framework_string(ml_framework):
    return " > ".join(s[0] for s in ml_framework.steps)


def _exponential_mean(x, x_prev):
    return x * 0.99 + x_prev * 0.01
