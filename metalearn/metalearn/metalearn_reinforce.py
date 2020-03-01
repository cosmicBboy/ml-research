"""Reinforce module for training the CASH controller."""

from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn

from . import utils, loggers
from .tracking import MetricsTracker

EPSILON = np.finfo(np.float32).eps.item()
CLIP_GRAD = 1.0

Losses = namedtuple(
    "Losses", ["total", "actor", "critic", "entropy", "grad_norm"])


class MetaLearnReinforce(object):
    """Reinforce component of deep-cash algorithm."""

    def __init__(
            self,
            controller,
            t_env,
            gamma=0.99,
            entropy_coef=0.0,
            entropy_coef_anneal_to=0.0,
            entropy_coef_anneal_by=None,
            normalize_reward=False,
            meta_reward_multiplier=1.,
            sample_new_task_every=10,
            metrics_logger=loggers.default_logger):
        """Initialize CASH Reinforce Algorithm.

        :param pytorch.nn.Module controller: A CASH controller to select
            actions.
        :param TaskEnvironment t_env: task environment to sample data
            environments and evaluate proposed ml frameworks.
        :param float gamma: discounted reward for computing returns.
        :param float entropy_coef: coefficient for entropy regularization.
        :param float entropy_coef: coefficient to anneal to by end of training.
        :param float entropy_coef_anneal_by: Value between 0.0 and 1.0
            indicating when during the course of n_episodes the entropy
            coefficient should reach 0. If None, no annealing is performed.
            For example if this value is 0.5, then the entropy coefficient will
            linearly reach 0.0 half-way through training.
        :param bool normalize_reward: whether or not to mean-center and
            standard-deviation-scale the reward signal for backprop.
        :param float meta_reward_multiplier: multiply the reward signal with
            this value. If this is set to 0, this effectively disables the
            metalearning reward signal.
        :param callable metrics_logger: logging function to use. The function
            takes as input a tracking.TrackerBase object and prints out a
            message.
        """
        self.controller = controller
        self.t_env = t_env
        self._gamma = gamma
        self._entropy_coef = entropy_coef
        self._entropy_coef_anneal_to = entropy_coef_anneal_to
        self._entropy_coef_anneal_by = entropy_coef_anneal_by
        self._normalize_reward = normalize_reward
        self._meta_reward_multiplier = meta_reward_multiplier
        self._metrics_logger = metrics_logger

        if entropy_coef_anneal_by is not None and \
                not 0 < entropy_coef_anneal_by <= 1:
            raise ValueError(
                "entropy_coef_anneal must be between (0, 1], found %s" %
                entropy_coef_anneal_by)

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

        # set entropy coefficient schedule per episode
        if self._entropy_coef_anneal_by is not None:
            episodes_to_zero = int(n_episodes * self._entropy_coef_anneal_by)
            # linearly decrease coefficient to zero
            self._entropy_coef_schedule = np.linspace(
                self._entropy_coef,
                self._entropy_coef_anneal_to,
                num=episodes_to_zero).tolist()
            self._entropy_coef_schedule += [
                self._entropy_coef_anneal_to
                for _ in range(n_episodes - episodes_to_zero)]
        else:
            self._entropy_coef_schedule = [self._entropy_coef] * n_episodes

        # track metrics
        self.tracker = MetricsTracker()

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
                i_episode + 1, self.t_env.current_data_env.name)
            if procnum is not None:
                msg = "proc num: %d, %s" % (procnum, msg)
            print("\n" + msg)

            last_value = self.start_episode(n_iter, verbose)

            # accumulate stats
            n_successful_mlfs = len(self._successful_mlfs)
            n_unique_mlfs = len(set((tuple(s) for s in self._algorithm_sets)))
            n_unique_hyperparams = len(set(
                [str(d.items()) for d in self._hyperparameter_sets]))
            mlf_diversity = utils._diversity_metric(
                n_unique_mlfs, n_successful_mlfs)
            hyperparam_diversity = utils._diversity_metric(
                n_unique_hyperparams, n_successful_mlfs)

            mean_rewards = np.mean(self.controller.reward_buffer)
            if len(self._validation_scores) > 0:
                mean_val_scores = np.mean(self._validation_scores)
                std_val_scores = np.std(self._validation_scores)
            else:
                mean_val_scores, std_val_scores = np.nan, np.nan

            # backward pass through controller
            losses = self.end_episode(
                last_value, self._entropy_coef_schedule[i_episode])

            self.tracker.update_metrics({
                "episode": i_episode + 1,
                "data_env_names": self.t_env.current_data_env.name,
                "target_type": self.t_env.current_data_env.target_type.name,
                "scorers": self.t_env.scorer.name,
                "total_losses": losses.total,
                "actor_losses": losses.actor,
                "critic_losses": losses.critic,
                "entropy_losses": losses.entropy,
                "gradient_norms": losses.grad_norm,
                "mean_rewards": mean_rewards,
                "mean_validation_scores": mean_val_scores,
                "std_validation_scores": std_val_scores,
                "best_validation_scores": self._best_validation_score,
                "best_mlfs": self._best_mlf,
                "n_successful_mlfs": n_successful_mlfs,
                "n_unique_mlfs": n_unique_mlfs,
                "n_unique_hyperparams": n_unique_hyperparams,
                "mlf_diversity": mlf_diversity,
                "hyperparam_diversity": hyperparam_diversity,
                "entropy_coefficient": self._entropy_coef_schedule[i_episode],
            })
            if self._metrics_logger is not None:
                self._metrics_logger(self.tracker)
        return self

    @property
    def history(self):
        """Get metrics history per episode."""
        return self.tracker.history

    @property
    def best_mlfs(self):
        """Get best mlfs per episode."""
        return self.tracker.best_mlfs

    def start_episode(self, n_iter, verbose):
        """Begin training the controller."""
        self._n_valid_mlf = 0
        prev_reward, prev_action = 0, self.controller.init_action()
        prev_hidden = self.controller.init_hidden()
        for i in range(n_iter):
            state = self.t_env.sample_task_state()
            prev_reward, prev_action, prev_hidden = self._fit_iter(
                state,
                self.t_env.current_data_env.target_type,
                prev_reward * self._meta_reward_multiplier,
                prev_action,
                prev_hidden)
            if verbose:
                print(
                    "iter %d - n valid mlf: %d/%d" % (
                        i, self._n_valid_mlf, i + 1),
                    sep=" ", end="\r", flush=True)

        state = self.t_env.sample_task_state()
        last_value, *_ = self.controller(
            prev_action=prev_action,
            prev_reward=utils.aux_tensor(prev_reward),
            metafeatures=state,
            hidden=prev_hidden,
            target_type=self.t_env.current_data_env.target_type,
        )
        return last_value.detach().item()

    def _fit_iter(
            self,
            metafeature_tensor,
            target_type,
            prev_reward,
            prev_action,
            prev_hidden):
        value, actions, action_activation, hidden = self.controller(
            prev_action=prev_action,
            prev_reward=utils.aux_tensor(prev_reward),
            metafeatures=metafeature_tensor,
            hidden=prev_hidden,
            target_type=target_type,
        )
        reward = self.evaluate_actions(actions, action_activation)

        self.controller.value_buffer.append(value)
        self.controller.log_prob_buffer.append(
            [a["log_prob"] for a in actions])
        self.controller.reward_buffer.append(reward)
        self.controller.entropy_buffer.append([a["entropy"] for a in actions])

        return reward, action_activation, hidden

    def evaluate_actions(self, actions, action_activation):
        """Evaluate actions on the validation set of the data environment."""
        algorithms, hyperparameters = utils.get_mlf_components(actions)
        mlf = self.controller.a_space.create_ml_framework(
            algorithms, hyperparameters=hyperparameters,
            task_metadata=self.t_env.get_current_task_metadata())
        mlf, reward, score = self.t_env.evaluate(mlf)
        self._action_buffer.append(action_activation)
        if reward is None:
            return self.t_env.error_reward
        else:
            self._validation_scores.append(score)
            self._n_valid_mlf += 1
            self._successful_mlfs.append(mlf)
            self._algorithm_sets.append(algorithms)
            self._hyperparameter_sets.append(hyperparameters)
            if self._best_validation_score is None or \
                    self.t_env.scorer.comparator(
                        score, self._best_validation_score):
                self._best_validation_score = score
                self._best_mlf = mlf
            return reward

    def end_episode(self, last_value, entropy_coef):
        """End an episode with one backpropagation step.

        Since REINFORCE algorithm is a gradient ascent method, negate the log
        probability in order to do gradient descent on the negative expected
        rewards.

        :returns: tuple of loss and aggregated gradient.
        :rtype: tuple[float, float]
        """
        _check_buffers(
            self.controller.value_buffer,
            self.controller.log_prob_buffer,
            self.controller.reward_buffer,
            self.controller.entropy_buffer)

        n = len(self.controller.reward_buffer)

        # reset gradient
        self.optim.zero_grad()

        # compute Q values
        returns = torch.zeros(len(self.controller.value_buffer))
        R = last_value
        for t in reversed(range(n)):
            R = self.controller.reward_buffer[t] + self._gamma * R
            returns[t] = R

        # mean-center and std-scale returns
        if self._normalize_reward:
            returns = (returns - returns.mean()) / (returns.std() + EPSILON)

        values = torch.cat(self.controller.value_buffer).squeeze()
        advantage = returns - values

        # compute loss
        actor_loss = [
            -l * a
            for log_probs, a in zip(self.controller.log_prob_buffer, advantage)
            for l in log_probs
        ]
        actor_loss = torch.cat(actor_loss).mean()
        critic_loss = 0.5 * advantage.pow(2).mean()

        # entropy loss term, negate the mean since we want to maximize entropy
        entropies = torch.cat([
            e * entropy_coef for entropy_list in self.controller.entropy_buffer
            for e in entropy_list
        ]).squeeze()
        entropy_loss = -entropies.mean()

        actor_critic_loss = actor_loss + critic_loss + entropy_loss

        # one step of gradient descent
        actor_critic_loss.backward()
        # gradient clipping to prevent exploding gradient
        nn.utils.clip_grad_norm_(self.controller.parameters(), CLIP_GRAD)
        self.optim.step()

        grad_norm = 0.
        for p in self.controller.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                grad_norm += param_norm.item() ** 2
        grad_norm = grad_norm ** 0.5

        # reset rewards and log probs
        del self.controller.value_buffer[:]
        del self.controller.log_prob_buffer[:]
        del self.controller.reward_buffer[:]
        del self.controller.entropy_buffer[:]

        return Losses(
            actor_critic_loss.data.item(),
            actor_loss.data.item(),
            critic_loss.data.item(),
            entropy_loss.data.item(),
            grad_norm)


def _check_buffers(
        value_buffer, log_prob_buffer, reward_buffer, entropy_buffer):
    n_vbuf = len(value_buffer)
    n_abuf = len(log_prob_buffer)
    n_rbuf = len(reward_buffer)
    n_ebuf = len(entropy_buffer)
    if n_vbuf != n_abuf != n_rbuf != n_ebuf:
        raise ValueError(
            "all buffers need the same length, found:\n"
            "value_buffer = %d\n"
            "log_prob_buffer = %d\n"
            "reward_buffer = %d\n"
            "entropy_buffer = %d\n" % (n_abuf, n_rbuf, n_ebuf))
