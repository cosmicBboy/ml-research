"""Reinforce module for training the CASH controller."""

import numpy as np
import torch
import torch.nn as nn

from collections import defaultdict
from torch.autograd import Variable

from . import utils

SINGLE_BASELINE = "all_data_envs"
EPSILON = np.finfo(np.float32).eps.item()


class CASHReinforce(object):
    """Reinforce component of deep-cash algorithm."""

    def __init__(
            self,
            controller,
            t_env,
            beta=0.99,
            entropy_coef=0.0,
            with_baseline=True,
            single_baseline=True,
            normalize_reward=True,
            metrics_logger=None):
        """Initialize CASH Reinforce Algorithm.

        :param pytorch.nn.Module controller: A CASH controller to select
            actions.
        :param TaskEnvironment t_env: task environment to sample data
            environments and evaluate proposed ml frameworks.
        :param float beta: hyperparameter for exponential moving average to
            compute baseline reward (used to regularize REINFORCE).
        :param float entropy_coef: coefficient for entropy regularization.
        :param bool with_baseline: whether or not to regularize the controller
            with the exponential moving average of past rewards.
        :param bool single_baseline: if True, maintains a single baseline
            reward buffer, otherwise maintain a separate baseline buffer for
            each data environment available to the task environment.
        :param bool normalize_reward: whether or not to mean-center and
            standard-deviation-scale the reward signal for backprop.
        :param callable metrics_logger: loggin function to use. The function
            takes as input a CASHReinforce object and prints out a message,
            with access to all properties in CASHReinforce.
        :ivar defaultdict[str -> list] _baseline_fn: a map that keeps track
            of the baseline function for a particular task environment state
            (i.e. the sampled data environment at a particular episode).
        """
        self.controller = controller
        self.t_env = t_env
        self._beta = beta
        self._entropy_coef = entropy_coef
        self._with_baseline = with_baseline
        self._single_baseline = single_baseline
        self._normalize_reward = normalize_reward
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
        self.aggregate_gradients = []
        self.mean_validation_scores = []
        self.std_validation_scores = []
        self.n_successful_mlfs = []
        self.n_unique_mlfs = []
        self.n_unique_hyperparams = []
        self.mlf_diversity = []
        self.hyperparam_diversity = []
        self.best_validation_scores = []
        self.best_mlfs = []

        # baseline reward
        self._baseline_fn = {
            "buffer": defaultdict(list),
            "current": defaultdict(int),
        }
        # the buffer history keeps track of baseline rewards across all the
        # episodes, used for record-keeping and testing purposes.
        self._baseline_buffer_history = defaultdict(list)

        for i_episode in range(self._n_episodes):
            self.t_env.sample_data_env()
            self._action_buffer = []
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
            # since the action sequences within episode are no longer
            # independent, need to get a single baseline value (the
            # exponential mean of the return for the previous episode)
            buffer = self._baseline_buffer()
            if len(buffer) == 0:
                baseline = self._baseline_current()
            else:
                baseline = buffer[-1]
                del self._baseline_buffer()[:]

            self.start_episode(n_iter, verbose)

            # accumulate stats
            n_successful_mlfs = len(self._successful_mlfs)
            n_unique_mlfs = len(set((tuple(s) for s in self._algorithm_sets)))
            n_unique_hyperparams = len(set(
                [str(d.items()) for d in self._hyperparameter_sets]))
            mlf_diversity = utils._diversity_metric(
                n_unique_mlfs, n_iter)
            hyperparam_diversity = utils._diversity_metric(
                n_unique_hyperparams, n_iter)
            self.data_env_names.append(self.t_env.data_env_name)
            self.mean_rewards.append(np.mean(self.controller.reward_buffer))
            if len(self._validation_scores) > 0:
                self.mean_validation_scores.append(
                    np.mean(self._validation_scores))
                self.std_validation_scores.append(
                    np.std(self._validation_scores))
            else:
                self.mean_validation_scores.append(np.nan)
                self.std_validation_scores.append(np.nan)
            self.n_successful_mlfs.append(n_successful_mlfs)
            self.n_unique_mlfs.append(n_unique_mlfs)
            self.n_unique_hyperparams.append(n_unique_hyperparams)
            self.mlf_diversity.append(mlf_diversity)
            self.hyperparam_diversity.append(hyperparam_diversity)
            self.best_validation_scores.append(self._best_validation_score)
            self.best_mlfs.append(self._best_mlf)

            # backward pass through controller
            loss, grad_agg = self.end_episode(
                baseline, mlf_diversity, hyperparam_diversity)
            self.losses.append(loss)
            self.aggregate_gradients.append(grad_agg)
            if self._metrics_logger:
                self._metrics_logger(self)
            else:
                print(
                    "\nloss: %0.02f - "
                    "mean performance: %0.02f - "
                    "mean reward: %0.02f - "
                    "grad agg: %0.02f - "
                    "mlf diversity: %d/%d" % (
                        self.losses[-1],
                        self.mean_validation_scores[-1],
                        self.mean_rewards[-1],
                        self.aggregate_gradients[-1],
                        self.n_unique_mlfs[-1],
                        self.n_successful_mlfs[-1])
                )
        return self

    def history(self):
        """Get metadata history."""
        return {
            "episode": range(1, self._n_episodes + 1),
            "data_env_names": self.data_env_names,
            "losses": self.losses,
            "aggregate_gradients": self.aggregate_gradients,
            "mean_rewards": self.mean_rewards,
            "mean_validation_scores": self.mean_validation_scores,
            "std_validation_scores": self.std_validation_scores,
            "n_successful_mlfs": self.n_successful_mlfs,
            "n_unique_mlfs": self.n_unique_mlfs,
            "n_unique_hyperparams": self.n_unique_hyperparams,
            "mlf_diversity": self.mlf_diversity,
            "hyperparam_diversity": self.hyperparam_diversity,
            "best_validation_scores": self.best_validation_scores,
            "best_mlfs": self.best_mlfs,
        }

    def start_episode(self, n_iter, verbose):
        """Begin training the controller."""
        self._n_valid_mlf = 0
        prev_reward, prev_action = 0, self.controller.init_action()
        for i_iter in range(n_iter):
            prev_reward, prev_action = self._fit_iter(
                self.t_env.sample(), self.t_env.target_type, prev_reward,
                prev_action)
            if verbose:
                print(
                    "iter %d - n valid mlf: %d/%d" % (
                        i_iter, self._n_valid_mlf, i_iter + 1),
                    sep=" ", end="\r", flush=True)

    def _fit_iter(
            self, metafeature_tensor, target_type, prev_reward, prev_action):
        actions, action_activation = self.controller.decode(
            init_input_tensor=prev_action,
            target_type=target_type,
            aux=aux_tensor(prev_reward),
            metafeatures=Variable(metafeature_tensor),
            init_hidden=self.controller.init_hidden())
        reward = self.evaluate_actions(actions, action_activation)
        self._update_log_prob_buffer(actions)
        self._update_reward_buffer(reward)
        self._update_entropy_buffer(actions)
        self._update_baseline_reward_buffer(reward)
        for k, v in self._baseline_fn["buffer"].items():
            self._baseline_buffer_history[k].extend(v)
        return reward, Variable(action_activation.data)

    def evaluate_actions(self, actions, action_activation):
        """Evaluate actions on the validation set of the data environment."""
        algorithms, hyperparameters = self._get_mlf_components(actions)
        mlf = self.controller.a_space.create_ml_framework(
            algorithms, hyperparameters=hyperparameters,
            env_dep_hyperparameters=self.t_env.env_dep_hyperparameters())
        reward, score = self.t_env.evaluate(mlf)
        self._action_buffer.append(action_activation)
        self._algorithm_sets.append(algorithms)
        self._hyperparameter_sets.append(hyperparameters)
        if reward is None:
            return self.t_env.error_reward
        else:
            self._validation_scores.append(score)
            self._n_valid_mlf += 1
            self._successful_mlfs.append(mlf)
            if self._best_validation_score is None or \
                    score > self._best_validation_score:
                self._best_validation_score = score
                self._best_mlf = mlf
            return reward

    def end_episode(
            self,
            baseline,
            mlf_diversity,
            h_diversity):
        """End an episode with one backpropagation step.

        Since REINFORCE algorithm is a gradient ascent method, negate the log
        probability in order to do gradient descent on the negative expected
        rewards.

        :param float baseline: baseline reward value, the exponential mean of
            rewards over previous episodes.
        :param float mlf_diversity: 1.0 means all successfully fitted mlfs
            are unique, 0.0 if there is only a single unique mlf in the batch.
        :param float h_diversity: 1.0 means all successfully
            fitted mlfs are have unique hyperparameters, 0.0 if there is only a
            single unique hyperparameter setting in the batch.
        :returns: tuple of loss and aggregated gradient.
        :rtype: tuple[float, float]
        """
        _check_buffers(
            self.controller.log_prob_buffer,
            self.controller.reward_buffer,
            self.controller.entropy_buffer)
        self.optim.zero_grad()
        n = len(self.controller.reward_buffer)

        R = torch.FloatTensor(self.controller.reward_buffer)
        if self._with_baseline:
            R = R - baseline
        if self._normalize_reward:
            R = normalize_reward(R)

        # compute loss
        loss = [-l * r - self._entropy_coef * e
                for log_probs, r, entropies in zip(
                    self.controller.log_prob_buffer, R,
                    self.controller.entropy_buffer)
                for l, e in zip(log_probs, entropies)]

        loss = torch.cat(loss).sum().div(n)

        # one step of gradient descent
        loss.backward()
        # gradient clipping to prevent exploding gradient
        nn.utils.clip_grad_norm(self.controller.parameters(), 20)
        self.optim.step()

        grad_agg = sum([
            p.grad.data.sum() for p in self.controller.parameters()
            if p.grad is not None])

        # reset rewards and log probs
        del self.controller.log_prob_buffer[:]
        del self.controller.reward_buffer[:]
        del self.controller.entropy_buffer[:]

        return loss.data[0], grad_agg

    def _update_log_prob_buffer(self, actions):
        self.controller.log_prob_buffer.append(
            [a["log_prob"] for a in actions])

    def _update_reward_buffer(self, reward):
        self.controller.reward_buffer.append(reward)

    def _update_entropy_buffer(self, actions):
        self.controller.entropy_buffer.append([a["entropy"] for a in actions])

    def _update_baseline_reward_buffer(self, reward):
        buffer = self._baseline_buffer()
        current = self._baseline_current()
        buffer.append(current)
        self._update_current_baseline(
            utils._exponential_mean(reward, current, beta=self._beta))

    def _baseline_fn_key(self):
        return SINGLE_BASELINE if self._single_baseline else \
            self.t_env.data_env_name

    def _baseline_buffer(self):
        return self._baseline_fn["buffer"][self._baseline_fn_key()]

    def _baseline_current(self):
        return self._baseline_fn["current"][self._baseline_fn_key()]

    def _update_current_baseline(self, new_baseline):
        self._baseline_fn["current"][self._baseline_fn_key()] = new_baseline

    def _get_mlf_components(self, actions):
        algorithms = []
        hyperparameters = {}
        for action in actions:
            if action["action_type"] == self.controller.ALGORITHM:
                algorithms.append(action["action"])
            if action["action_type"] == self.controller.HYPERPARAMETER:
                hyperparameters[action["action_name"]] = action["action"]
        return algorithms, hyperparameters


def _check_buffers(log_prob_buffer, reward_buffer, entropy_buffer):
    n_abuf = len(log_prob_buffer)
    n_rbuf = len(reward_buffer)
    n_ebuf = len(entropy_buffer)
    if n_abuf != n_rbuf != n_ebuf:
        raise ValueError(
            "all buffers need the same length, found:\n"
            "log_prob_buffer = %d\n"
            "reward_buffer = %d\n"
            "entropy_buffer = %d\n" % (n_abuf, n_rbuf, n_ebuf))


def normalize_reward(rbuffer):
    """Mean-center and std-rescale the reward buffer.

    :param torch.FloatTensor rbuffer: list of rewards from an episode
    :returns: normalized tensor
    """
    return (rbuffer - rbuffer.mean()).div(rbuffer.std() + EPSILON)


def aux_tensor(prev_reward):
    """Create an auxiliary input tensor for previous reward and action.

    This is just the reward from the most recent iteration. At the beginning
    of each episode, the previous reward is reset to 0.
    """
    r_tensor = torch.zeros(1, 1, 1)
    r_tensor += prev_reward
    return Variable(r_tensor)


def print_actions(actions):
    """Pretty print actions for an ml framework."""
    for i, action in enumerate(actions):
        print("action #: %s" % i)
        for k, v in action.items():
            print("%s: %s" % (k, v))
