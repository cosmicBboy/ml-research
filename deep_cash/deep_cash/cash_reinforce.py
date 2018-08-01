"""Reinforce module for training the CASH controller."""

import numpy as np
import torch
import torch.nn as nn

from torch.autograd import Variable

from . import utils


class CASHReinforce(object):
    """Reinforce component of deep-cash algorithm."""

    def __init__(self, controller, t_env, beta=0.99,
                 with_baseline=True, metrics_logger=None):
        """Initialize CASH Reinforce Algorithm.

        :param pytorch.nn.Module controller: A CASH controller to select
            actions.
        :param TaskEnvironment t_env: task environment to sample data
            environments and evaluate proposed ml frameworks.
        :param float beta: hyperparameter for exponential moving average to
            compute baseline reward (used to regularize REINFORCE).
        :param bool with_baseline: whether or not to regularize the controller
            with the exponential moving average of past rewards.
        :param callable metrics_logger: loggin function to use. The function
            takes as input a CASHReinforce object and prints out a message,
            with access to all properties in CASHReinforce.
        """
        self.controller = controller
        self.t_env = t_env
        self._logger = utils.init_logging(__file__)
        self._beta = beta
        self._with_baseline = with_baseline
        self._metrics_logger = metrics_logger

    def fit(self, optim, optim_kwargs, n_episodes=100, n_iter=100,
            verbose=True, procnum=None):
        """Fits the CASH controller with the REINFORCE algorithm.

        :param torch.optim optim: type of optimization to use for backprop.
        :param dict optim_kwargs: arguments to pass to optim constructor
        :param int n_episodes: number of episodes to train.
        :param int n_iter: number of iterations per episode.
        :param bool verbose: whether or not to print the exponential mean
            reward per iteration.
        :param int procnum: optional argument to indicate the process number
            used for multiprocessing.
        """
        # set optimization object
        self.optim = optim(self.controller.parameters(), **optim_kwargs)
        self._n_episodes = n_episodes

        # track metrics
        self.data_env_names = []
        self.losses = []
        self.mean_rewards = []
        self.mean_validation_scores = []
        self.std_validation_scores = []
        self.n_successful_mlfs = []
        self.n_unique_mlfs = []
        self.n_unique_hyperparameters = []
        self.mlf_framework_diversity = []
        self.best_validation_scores = []
        self.best_mlfs = []

        # baseline reward
        self._current_baseline_reward = 0
        for i_episode in range(self._n_episodes):
            self.t_env.sample_data_env()
            self._validation_scores = []
            self._successful_mlfs = []
            self._algorithm_sets = []
            self._hyperparameter_sets = []
            self._best_validation_score = None
            self._best_mlf = None
            msg = "episode %d, task: %s" % (
                i_episode, self.t_env.data_env_name)
            if procnum is not None:
                msg = "proc num: %d, %s" % (procnum, msg)
            print("\n" + msg)
            self._fit_episode(n_iter, verbose)

            # episode stats
            mean_reward = np.mean(self.controller.reward_buffer)
            loss = self.backward(with_baseline=self._with_baseline)
            if len(self._validation_scores) > 0:
                mean_validation_score = np.mean(self._validation_scores)
                std_validation_score = np.std(self._validation_scores)
            else:
                mean_validation_score = np.nan
                std_validation_score = np.nan
            n_successful_mlfs = len(self._successful_mlfs)
            n_unique_mlfs = len(set((tuple(s) for s in self._algorithm_sets)))
            n_unique_hyperparameters = len(set(
                [str(d.items()) for d in self._hyperparameter_sets]))
            # accumulate stats
            # TODO: track unique hyperparameter settings in order to compute
            # number of unique hyperparameter settings, and number of unique
            # hyperparameter settings per mlf pipeline.
            self.data_env_names.append(self.t_env.data_env_name)
            self.losses.append(loss)
            self.mean_rewards.append(mean_reward)
            self.mean_validation_scores.append(mean_validation_score)
            self.std_validation_scores.append(std_validation_score)
            self.n_successful_mlfs.append(n_successful_mlfs)
            self.n_unique_mlfs.append(n_unique_mlfs)
            self.n_unique_hyperparameters.append(n_unique_hyperparameters)
            self.mlf_framework_diversity.append(
                utils._ml_framework_diversity(
                    n_unique_mlfs, n_successful_mlfs))
            self.best_validation_scores.append(self._best_validation_score)
            self.best_mlfs.append(self._best_mlf)
            if self._metrics_logger:
                self._metrics_logger(self)
            else:
                print(
                    "\nloss: %0.02f - "
                    "mean performance: %0.02f - "
                    "mean reward: %0.02f - "
                    "mlf diversity: %d/%d" % (
                        loss,
                        mean_validation_score,
                        mean_reward,
                        n_unique_mlfs,
                        n_successful_mlfs)
                )
        return self

    def history(self):
        """Get metadata history."""
        return {
            "episode": range(1, self._n_episodes + 1),
            "data_env_names": self.data_env_names,
            "losses": self.losses,
            "mean_rewards": self.mean_rewards,
            "mean_validation_scores": self.mean_validation_scores,
            "std_validation_scores": self.std_validation_scores,
            "n_successful_mlfs": self.n_successful_mlfs,
            "n_unique_mlfs": self.n_unique_mlfs,
            "n_unique_hyperparameters": self.n_unique_hyperparameters,
            "best_validation_scores": self.best_validation_scores,
            "best_mlfs": self.best_mlfs,
        }

    def _fit_episode(self, n_iter, verbose):
        self._n_valid_mlf = 0
        for i_iter in range(n_iter):
            self._fit_iter(self.t_env.sample())
            if verbose:
                print(
                    "iter %d - n valid mlf: %d/%d - "
                    "exponential mean reward: %0.02f" % (
                        i_iter, self._n_valid_mlf, i_iter + 1,
                        self._current_baseline_reward),
                    sep=" ", end="\r", flush=True)

    def _fit_iter(self, t_state):
        actions = self.select_actions(
            metafeature_tensor(t_state), self.controller.init_hidden())
        reward = self.evaluate_actions(actions)
        self._update_log_prob_buffer(actions)
        self._update_reward_buffer(reward)
        self._update_baseline_reward_buffer(reward)

    def backward(self, show_grad=False, with_baseline=True):
        """End an episode with one backpropagation step.

        Since REINFORCE algorithm is a gradient ascent method, negate the log
        probability in order to do gradient descent on the negative expected
        rewards.
        - If reward is positive and selected action prob is close to 1
          (-log prob ~= 0), policy loss will be positive and close to 0,
          controller's weights will be adjusted by gradients such that the
          selected action is more likely and the non-selected actions are less
          likely.
        - If reward is positive and selection action prob is close to 0
          (-log prob > 0), policy loss will be a large positive number,
          controller's weights will be adjusted such that selected
          action is less likely and non-selected actions are more likely.
        - If reward is negative and selected action prob is close to 1
          (-log prob ~= 0), policy loss will be negative but close to 0,
          meaning that gradients will adjust the weights to minimize policy
          loss (make it more negative) by making the selected action
          probability even more negative (push the selected action prob closer
          to 0, which discourages the selection of this action).
        - If reward is negative and selected action prob is close to 0,
          (-log prob > 0), then the policy loss will be a large negative number
          and the gradients will make the selected action prob even less likely
        """
        loss = []
        self._check_buffers()
        # compute loss
        for log_probs, r, b in zip(
                self.controller.log_prob_buffer,
                self.controller.reward_buffer,
                self.controller.baseline_reward_buffer):
            for a in log_probs:
                r = r - b if with_baseline else r
                loss.append(-a * r)

        # one step of gradient descent
        self.optim.zero_grad()
        loss = torch.cat(loss).sum().div(len(self.controller.reward_buffer))
        loss.backward()
        # gradient clipping to prevent exploding gradient
        nn.utils.clip_grad_norm(self.controller.parameters(), 20)
        self.optim.step()

        if show_grad:
            print("\n\nGradients")
            for param in self.controller.parameters():
                print(param.grad.data.sum())

        # reset rewards and log probs
        del self.controller.reward_buffer[:]
        del self.controller.log_prob_buffer[:]
        del self.controller.baseline_reward_buffer[:]

        return loss.data[0]

    def select_actions(self, init_input_tensor, init_hidden):
        """Select action sequence given initial input and hidden tensors."""
        actions = self.controller.decode(init_input_tensor, init_hidden)
        return actions

    def evaluate_actions(self, actions):
        """Evaluate actions on the validation set of the data environment."""
        algorithms, hyperparameters = self._get_mlf_components(actions)
        mlf = self.controller.a_space.create_ml_framework(
            algorithms, hyperparameters=hyperparameters,
            env_dep_hyperparameters=self.t_env.env_dep_hyperparameters())
        reward = self.t_env.evaluate(mlf)
        if reward is None:
            return self.t_env.error_reward
        else:
            self._n_valid_mlf += 1
            self._validation_scores.append(reward)
            self._successful_mlfs.append(mlf)
            self._algorithm_sets.append(algorithms)
            self._hyperparameter_sets.append(hyperparameters)
            if self._best_validation_score is None or \
                    reward > self._best_validation_score:
                self._best_validation_score = reward
                self._best_mlf = mlf
            return reward

    def _update_log_prob_buffer(self, actions):
        self.controller.log_prob_buffer.append(
            [a["action_log_prob"] for a in actions])

    def _update_reward_buffer(self, reward):
        self.controller.reward_buffer.append(reward)

    def _update_baseline_reward_buffer(self, reward):
        self.controller.baseline_reward_buffer.append(
            self._current_baseline_reward)
        # don't let the baseline_reward buffer exceed the length of the reward
        # buffer.
        if len(self.controller.baseline_reward_buffer) > \
                len(self.controller.reward_buffer):
            self.controller.baseline_reward_buffer.pop(0)
        self._current_baseline_reward = utils._exponential_mean(
            reward, self._current_baseline_reward, beta=self._beta)

    def _get_mlf_components(self, actions):
        algorithms = []
        hyperparameters = {}
        for action in actions:
            if action["action_type"] == self.controller.ALGORITHM:
                algorithms.append(action["action"])
            if action["action_type"] == self.controller.HYPERPARAMETER:
                hyperparameters[action["action_name"]] = action["action"]
        return algorithms, hyperparameters

    def _check_buffers(self):
        n_abuf = len(self.controller.log_prob_buffer)
        n_rbuf = len(self.controller.reward_buffer)
        n_bbuf = len(self.controller.baseline_reward_buffer)
        if n_abuf != n_rbuf != n_bbuf:
            raise ValueError(
                "all buffers need the same length, found:\n"
                "log_prob_buffer = %d\n"
                "reward_buffer = %d\n"
                "baseline_reward_buffer = %d" % (
                    n_abuf, n_rbuf, n_bbuf))


def metafeature_tensor(t_state):
    """Convert a metafeature vector into a tensor.

    For now this will be a single category indicating `is_executable`.

    :returns Tensor: dim <string_length x 1 x metafeature_dim>, where
        metafeature_dim is a continuous feature.
    """
    return Variable(utils._create_metafeature_tensor(t_state, [None]))


def print_actions(actions):
    """Pretty print actions for an ml framework."""
    for i, action in enumerate(actions):
        print("action #: %s" % i)
        for k, v in action.items():
            print("%s: %s" % (k, v))
