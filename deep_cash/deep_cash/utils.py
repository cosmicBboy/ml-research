"""Utility functions."""

import numpy as np

import torch
from torch.autograd import Variable

# specify type of the metafeature
METAFEATURES = [
    ("number_of_examples", int)
]


class PerformanceTracker(object):
    """Tracks performance during model fitting."""

    def __init__(self, num_candidates):
        self.num_candidates = num_candidates
        self.best_candidates = []
        self.best_scores = []
        self.overall_mean_reward = []
        self.overall_a_loss = []
        self.overall_h_loss = []
        self.overall_ml_score = []
        self.running_reward = 10

    def reset_episode(self):
        self.ml_score = []
        self.valid_frameworks = []

    def update_performance(self, rewards, i):
        self.running_reward = _exponential_mean(self.running_reward, i)
        self.overall_mean_reward.append(np.mean(rewards))
        self.overall_ml_score.append(
            np.mean(self.ml_score) if len(self.ml_score) > 0 else np.nan)

    def print_end_episode(self, i_episode, ep_length):
        print(
            "\nEp%s | mean reward: %0.02f | mean perf: %0.02f | "
            "ep length: %d | running reward: %0.02f" % (
                i_episode, self.overall_mean_reward[i_episode],
                self.overall_ml_score[i_episode], ep_length,
                self.running_reward))

    def maintain_best_candidates(self, ml_framework, score):
        """Maintain the best candidates and their associated scores."""
        if len(self.best_candidates) < self.num_candidates:
            self.best_candidates.append(ml_framework)
            self.best_scores.append(score)
        else:
            min_index = self.best_scores.index(min(self.best_scores))
            if score > self.best_scores[min_index]:
                self.best_candidates[min_index] = ml_framework
                self.best_scores[min_index] = score


class ControllerFitTracker(object):

    def __init__(
            self, activate_h_controller, init_n_hyperparams,
            increase_n_hyperparam_by, increase_n_hyperparam_every):
        self.activate_h_controller = activate_h_controller
        self.n_hyperparams = init_n_hyperparams
        self.increase_n_hyperparam_by = increase_n_hyperparam_by
        self.increase_n_hyperparam_every = increase_n_hyperparam_every
        self.previous_baseline_reward = 0

    def reset_episode(self):
        self.n_valid_frameworks = 0
        self.n_valid_frameworks = 0
        self.n_valid_hyperparams = 0
        self.current_baseline_reward = 0
        self.successful_frameworks = []

    def update_n_hyperparams(self, i_episode):
        if i_episode > 0 and i_episode > self.activate_h_controller and \
                i_episode % self.increase_n_hyperparam_every == 0:
            self.n_hyperparams += self.increase_n_hyperparam_by

    def update_current_baseline_reward(self, reward):
        self.current_baseline_reward = _exponential_mean(
            reward, self.current_baseline_reward)

    def update_prev_baseline_reward(self):
        self.previous_baseline_reward = self.current_baseline_reward

    def print_fit_progress(self, i):
        print(
            "%d/%d valid frameworks, %d/%d valid hyperparams "
            "%d/%d successful frameworks" % (
                self.n_valid_frameworks, i + 1,
                self.n_valid_hyperparams, i + 1,
                len(self.successful_frameworks), i + 1),
            sep=" ", end="\r", flush=True)

    def print_end_episode(self, i_episode):
        if len(self.successful_frameworks) > 0:
            print("last ml framework sample: %s" %
                  _ml_framework_string(self.successful_frameworks[-1]))
            print("framework diversity: %d/%d" % (
                len(set([_ml_framework_string(f)
                         for f in self.successful_frameworks])),
                len(self.successful_frameworks)))
        if i_episode > self.activate_h_controller:
            print("n_hyperparams: %d" % self.n_hyperparams)


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
