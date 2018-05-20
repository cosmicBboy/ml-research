"""Module for generating algorithms sequences.

TODO:
- add time cost to fitting a proposed framework
"""

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict

from sklearn.base import clone
from torch.autograd import Variable
from torch.distributions import Categorical

from . import utils
from .algorithm_space import START_TOKEN


class MLFrameworkController(object):
    """Controller to generate machine learning frameworks."""

    def __init__(
            self, a_controller, h_controller, a_space, optim=None,
            optim_kwargs=None):
        self.a_controller = a_controller
        self.h_controller = h_controller
        self.a_space = a_space
        self.optim = optim
        self.optim_kwargs = {} if optim_kwargs is None else optim_kwargs
        if optim is not None:
            self.optim = optim(
                list(self.a_controller.parameters()) +
                list(self.h_controller.parameters()),
                **optim_kwargs)

    def select_algorithms(self, t_state):
        """Select ML algorithms from the algorithm controller given task state.

        The task state is implemented as the metafeatures associated with a
        particular training set metafeatures, in the context of bootstrap
        sampling where an ML framework is fit on the bootstrapped sample and
        the validation performance (reward) is computed on the out-of-bag
        sample.

        :param AlgorithmControllerRNN a_controller:
        :param list t_state:
        """
        # Note that the first dimension of the metafeatures and input tensor
        # are both 1 because we want to train a generative model that proposes
        # policies, which are sequences of actions, each action specifying one
        # aspect of a machine learning framework.
        metafeature_tensor, input_tensor = utils._create_training_data_tensors(
            self.a_space, t_state, [START_TOKEN])
        hidden = self.a_controller.init_hidden()
        # TODO: eventually remove component_probs so that the hyperparameter
        # controller doesn't need the flattened softmax output of the algorithm
        # controller
        ml_framework, component_probs, log_probs = [], [], []
        for i in range(self.a_space.N_COMPONENT_TYPES):
            # algorithm controller
            probs, hidden = self.a_controller(
                metafeature_tensor, input_tensor, hidden)
            m = Categorical(probs)
            action = m.sample()
            log_probs.append(m.log_prob(action))
            component = self.a_space.components[int(action)]
            component_probs.append(probs.data)
            ml_framework.append(component)
            input_tensor = Variable(
                utils._create_input_tensor(self.a_space, [component]))

        # save log probabilities per action
        self.a_controller.saved_log_probs.append(torch.cat(log_probs).sum())

        # create a flattened vector of the softmax output for all
        # algorithm components. This is used as input to the hyperparameter
        # controller. In the future we'll want to exchange information between
        # the two controllers via the hidden state.
        component_probs = torch.cat(component_probs, dim=1)
        component_probs = component_probs.view(1, 1, component_probs.shape[1])
        return ml_framework, component_probs, hidden

    def select_hyperparameters(
            self, t_state, component_probs, n_hyperparams, inner=False,
            share_hidden=None):
        """Select Hyperparameters."""
        metafeature_tensor = utils._create_metafeature_tensor(
            t_state, [START_TOKEN])
        metafeature_tensor = Variable(torch.cat(
            [metafeature_tensor, component_probs], dim=2))
        hyperparam_tensor = utils._create_hyperparameter_tensor(
            self.a_space, [self.a_space.h_start_token_index])
        hidden = self.h_controller.init_hidden() if share_hidden is None \
            else share_hidden
        log_probs, hyperparameters, value_indices = [], [], []
        # compute joint log probability of hyperparameters
        for h_name in self.a_space.hyperparameter_name_space[:n_hyperparams]:
            probs, hidden = self.h_controller(
                metafeature_tensor, hyperparam_tensor, hidden)
            m = Categorical(probs)
            action = m.sample()
            log_probs.append(m.log_prob(action))
            h_value = self.a_space.hyperparameter_state_space_values[
                int(action)]
            value_indices.append((h_name, int(action)))
            hyperparameters.append((h_name, h_value))
            hyperparam_tensor = utils._create_hyperparameter_tensor(
                self.a_space, [int(action)])
        log_prob_attr = "inner_saved_log_probs" if inner else "saved_log_probs"
        getattr(self.h_controller, log_prob_attr).append(
                torch.cat(log_probs).sum())
        return OrderedDict(hyperparameters), OrderedDict(value_indices)

    def select_eval_ml_framework(
            self, t_env, t_state, h_controller_is_active, sig_check,
            with_inner_hloop):
        """Select ML framework given task state.

        TODO: implement variable reward scheme for succeed at different parts
        of the task, e.g. creating an ml framework that passes
        `check_ml_framework`, `check_hyperparameters`, and `t_env.evaluate`
        so that the controller can learn the different aspects of the task.
        """
        # ml framework pipeline creation
        pipeline, component_probs, a_hidden = self.select_algorithms(t_state)
        ml_framework = self.a_space.check_ml_framework(
            pipeline, sig_check=sig_check)
        if ml_framework is None:
            # fail to create a valid pipeline of algorithms
            return t_env.error_reward

        self.cf_tracker.n_valid_frameworks += 1
        # hyperparameter controller inner and outer loop
        if h_controller_is_active:
            if with_inner_hloop:
                _ = self.fit_h_controller(
                    ml_framework, t_env, t_state, component_probs,
                    n_iter=1, n_hyperparams=self.cf_tracker.n_hyperparams)
            hyperparameters, h_value_indices = self.select_hyperparameters(
                t_state, component_probs,
                n_hyperparams=self.cf_tracker.n_hyperparams,
                share_hidden=a_hidden)
            ml_framework = self.a_space.check_hyperparameters(
                ml_framework, hyperparameters, h_value_indices)

        # ml framework evaluation
        if ml_framework is None:
            # create valid pipeline, but fail hyperparameter specification
            return t_env.error_reward + self.a_space.VALID_MLF_BONUS

        self.cf_tracker.n_valid_hyperparams += 1
        reward = t_env.evaluate(ml_framework)

        if reward is None:
            # create valid pipeline and hyperparams, fail to fit and evaluate
            # on task environment
            return t_env.error_reward + self.a_space.VALID_MLFH_BONUS

        self.cf_tracker.successful_frameworks.append(ml_framework)
        self.p_tracker.ml_score.append(reward)
        self.p_tracker.maintain_best_candidates(ml_framework, reward)
        return reward

    def backward(self, baseline_reward, show_grad=False):
        """Joint backward pass through hyperparam and algorithm controllers."""
        if self.optim is None:
            raise ValueError(
                "optimization object not set. You need to provide `optim` "
                "and `optim_kwargs` when instantiating an %s object" % self)
        loss = []

        a_log_probs = self.a_controller.saved_log_probs
        a_rewards = self.a_controller.rewards
        h_log_probs = self.h_controller.saved_log_probs
        h_rewards = self.h_controller.rewards

        log_probs_rewards = [(a_log_probs, a_rewards)]
        if len(h_log_probs) > 0:
            log_probs_rewards.append((h_log_probs, h_rewards))

        # compute loss
        for log_probs, rewards in log_probs_rewards:
            for log_prob, reward in zip(log_probs, rewards):
                loss.append(-log_prob * (reward - baseline_reward))

        # one step of gradient descent
        self.optim.zero_grad()
        loss = torch.cat(loss).sum().div(len(rewards))
        loss.backward()
        # gradient clipping to prevent exploding gradient
        nn.utils.clip_grad_norm(self.a_controller.parameters(), 20)
        nn.utils.clip_grad_norm(self.h_controller.parameters(), 20)
        self.optim.step()

        if show_grad:
            print("\n\nAlgorithm controller gradients")
            for param in self.a_controller():
                print(param.grad.data.sum())
            print("\n\nHyperparameter controller gradients")
            for param in self.h_controller():
                print(param.grad.data.sum())

        # reset rewards and log probs
        del a_log_probs[:]
        del a_rewards[:]
        del h_log_probs[:]
        del h_rewards[:]

        return loss.data[0]

    def fit_h_controller(
            self, ml_framework, t_env, t_state, component_probs, n_iter,
            n_hyperparams=1, verbose=False):
        """Train the hyperparameter controller given an ML framework.

        This is a separate inner loop for training the h_controller for a
        particular ml_framework.
        """
        curr_baseline_r, prev_baseline_r = 0, 0
        if verbose:
            print("\n%s" % utils._ml_framework_string(ml_framework))
        for h_i_episode in range(n_iter):
            n_valid_hyperparams = 0
            for i in range(100):
                mlf = clone(ml_framework)
                hyperparams, h_value_indices = self.select_hyperparameters(
                    t_state, component_probs, n_hyperparams, inner=True)
                mlf = self.a_space.check_hyperparameters(
                    mlf, hyperparams, h_value_indices)
                if mlf is None:
                    r = t_env.error_reward
                else:
                    n_valid_hyperparams += 1
                    r = t_env.correct_hyperparameter_reward
                self.h_controller.inner_rewards.append(r)
                curr_baseline_r = utils._exponential_mean(r, curr_baseline_r)
                if verbose:
                    print("Ep%d, %d/%d valid hyperparams %s%s" %
                          (h_i_episode, n_valid_hyperparams, i + 1,
                           h_value_indices, " " * 10),
                          sep=" ", end="\r", flush=True)
            inner_h_loss = self.h_controller.inner_backward(prev_baseline_r)
            prev_baseline_r = curr_baseline_r
        return inner_h_loss

    def fit(self, t_env, num_episodes=10, log_every=1, n_iter=1000,
            show_grad=False, num_candidates=10, activate_h_controller=2,
            init_n_hyperparams=1, increase_n_hyperparam_by=3,
            increase_n_hyperparam_every=5, sig_check_interval=25,
            with_inner_hloop=False):
        """Train the AlgorithmContoller RNN.

        :param DataSpace t_env: An environment that generates tasks to do on a
            particular dataset and evaluates the solution proposed by the
            algorithm controller.
        :param int init_n_hyperparams: starting number of hyperparameters to
            propose.
        :param int increase_n_hyperparam_by: increase number of hyperparameters
            by this much.
        :param int increase_n_hyperparam_every: increase number of
            hyperparameters to propose after this many episodes.
        :param int sig_check_interval: interval to increase the strictness
            of check_ml_framework. Starts with checking just one component,
            but then slowly increment to a_space.N_COMPONENT_TYPES.
        :param bool with_inner_hloop: train the hyperparameter controller with
            an inner loop, given a particular ml framework.
        """
        self.p_tracker = utils.PerformanceTracker(num_candidates)
        self.cf_tracker = utils.ControllerFitTracker(
            activate_h_controller, init_n_hyperparams,
            increase_n_hyperparam_by, increase_n_hyperparam_every)
        # sig check is the number of components to check
        sig_check = 1
        for i_episode in range(num_episodes):
            with utils.Timer() as episode_t:
                if i_episode > 0 and i_episode % sig_check_interval == 0 \
                        and sig_check < self.a_space.N_COMPONENT_TYPES:
                    sig_check += 1
                # sample data environment from data distribution
                t_env.sample_data_env()
                # sample training/test data from data environment
                t_state = t_env.sample()
                self.p_tracker.reset_episode()
                self.cf_tracker.reset_episode()
                self.cf_tracker.update_n_hyperparams(i_episode)

                for i in range(n_iter):
                    with utils.Timer() as eval_mlf_t:
                        reward = self.select_eval_ml_framework(
                            t_env, t_state, i_episode > activate_h_controller,
                            sig_check, with_inner_hloop)
                        self.a_controller.rewards.append(reward)
                        self.h_controller.rewards.append(reward)
                        self.cf_tracker.print_fit_progress(i)
                        self.cf_tracker.update_current_baseline_reward(reward)
                        t_state = t_env.sample()

                self.p_tracker.update_performance(self.a_controller.rewards, i)
                self.p_tracker.overall_h_loss.append(np.nan)

                # backward pass
                with utils.Timer() as a_backward_t:
                    a_loss = self.backward(
                        self.cf_tracker.previous_baseline_reward)
                self.p_tracker.overall_a_loss.append(a_loss)
                # update baseline rewards
                self.cf_tracker.update_prev_baseline_reward()
                if i_episode % log_every == 0:
                    self.p_tracker.print_end_episode(i_episode, i + 1)
                    self.cf_tracker.print_end_episode(i_episode)
            print("ep time: %.03f sec, algorithm backward time: %.03f sec, "
                  "eval mlf time: %.03f sec, sigcheck: %d" % (
                    episode_t.interval,
                    a_backward_t.interval,
                    eval_mlf_t.interval,
                    sig_check))
            print("")
        return self.p_tracker
