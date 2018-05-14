"""Module for generating algorithms sequences.

TODO:
- remove dependency of MLFrameworkController on AlgorithmSpace
- add time cost to fitting a proposed framework
"""

import numpy as np
import torch

from collections import OrderedDict

from sklearn.base import clone
from torch.autograd import Variable
from torch.distributions import Categorical

from . import utils
from .algorithm_space import START_TOKEN


class MLFrameworkController(object):
    """Controller to generate machine learning frameworks."""

    def __init__(self, a_controller, h_controller, a_space):
        self.a_controller = a_controller
        self.h_controller = h_controller
        self.a_space = a_space
        self.tracker = _performance_tracker()

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

        # save log probabilities per paction
        self.a_controller.saved_log_probs.append(torch.cat(log_probs).sum())

        # create a flattened vector of the softmax output for all
        # algorithm components. This is used as input to the hyperparameter
        # controller. In the future we'll want to exchange information between
        # the two controllers via the hidden state.
        component_probs = torch.cat(component_probs, dim=1)
        component_probs = component_probs.view(1, 1, component_probs.shape[1])
        return ml_framework, component_probs

    def select_hyperparameters(
            self, t_state, component_probs, n_hyperparams, inner=False):
        """Select Hyperparameters."""
        metafeature_tensor = utils._create_metafeature_tensor(
            t_state, [START_TOKEN])
        metafeature_tensor = Variable(torch.cat(
            [metafeature_tensor, component_probs], dim=2))
        hyperparam_tensor = utils._create_hyperparameter_tensor(
            self.a_space, [self.a_space.h_start_token_index])
        hidden = self.h_controller.init_hidden()
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

    def fit(self, t_env, num_episodes=10, log_every=1, n_iter=1000,
            show_grad=False, num_candidates=10, activate_h_controller=2,
            init_n_hyperparams=1, increase_n_hyperparam_by=3,
            increase_n_hyperparam_every=5):
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
        """
        prev_baseline_reward = 0
        n_hyperparams = init_n_hyperparams
        for i_episode in range(num_episodes):
            # sample data environment from data distribution
            t_env.sample_data_env()
            # sample training/test data from data environment
            t_state = t_env.sample()
            n_valid_frameworks, n_valid_hyperparams, n_successful = 0, 0, 0
            current_baseline_reward = 0
            ml_performance, valid_frameworks = [], []

            # increment number of hyperparameters to predict
            if i_episode > 0 and i_episode > activate_h_controller and \
                    i_episode % increase_n_hyperparam_every == 0:
                n_hyperparams += increase_n_hyperparam_by

            for i in range(n_iter):
                # ml framework pipeline creation
                pipeline, component_probs = self.select_algorithms(t_state)
                ml_framework = check_ml_framework(self.a_space, pipeline)
                if ml_framework is None:
                    reward = t_env.error_reward
                else:
                    n_valid_frameworks += 1

                    # hyperparameter controller inner and outer loop
                    if i_episode > activate_h_controller:
                        _ = self.fit_h_controller(
                            ml_framework, t_env, t_state, component_probs,
                            n_iter=1, n_hyperparams=n_hyperparams)
                        hyperparameters, h_value_indices = \
                            self.select_hyperparameters(
                                t_state, component_probs,
                                n_hyperparams=n_hyperparams)
                        ml_framework = check_hyperparameters(
                            ml_framework, self.a_space, hyperparameters,
                            h_value_indices)

                    # ml framework evaluation
                    if ml_framework is None:
                        reward = t_env.error_reward
                    else:
                        n_valid_hyperparams += 1
                        reward = t_env.evaluate(ml_framework)
                        if reward is None:
                            reward = t_env.error_reward
                        else:
                            valid_frameworks.append(ml_framework)
                            n_successful += 1
                            ml_performance.append(reward)
                            _maintain_best_candidates(
                                self.tracker, num_candidates, ml_framework,
                                reward)
                print("%d/%d valid frameworks, %d/%d valid hyperparams "
                      "%d/%d successful frameworks" %
                      (n_valid_frameworks, i + 1,
                       n_valid_hyperparams, i + 1,
                       n_successful, i + 1),
                      sep=" ", end="\r", flush=True)
                current_baseline_reward = utils._exponential_mean(
                    reward, current_baseline_reward)
                t_state = t_env.sample()
                self.a_controller.rewards.append(reward)
                self.h_controller.rewards.append(reward)

            self.tracker = self.tracker._replace(
                running_reward=utils._exponential_mean(
                    self.tracker.running_reward, i))
            self.tracker.overall_mean_reward.append(
                np.mean(self.a_controller.rewards))
            self.tracker.overall_ml_performance.append(
                np.mean(ml_performance) if len(ml_performance) > 0 else np.nan)

            if i_episode > activate_h_controller:
                h_loss = self.h_controller.backward(prev_baseline_reward)
                self.tracker.overall_h_loss.append(h_loss)
            else:
                self.tracker.overall_h_loss.append(np.nan)

            # backward pass
            a_loss = self.a_controller.backward(prev_baseline_reward)
            self.tracker.overall_a_loss.append(a_loss)
            # update baseline rewards
            prev_baseline_reward = current_baseline_reward
            if i_episode % log_every == 0:
                print("\nEp%s | mean reward: %0.02f | "
                      "mean perf: %0.02f | ep length: %d | "
                      "running reward: %0.02f" %
                      (i_episode, self.tracker.overall_mean_reward[i_episode],
                       self.tracker.overall_ml_performance[i_episode], i + 1,
                       self.tracker.running_reward))
                if len(valid_frameworks) > 0:
                    print("last ml framework sample: %s" %
                          utils._ml_framework_string(valid_frameworks[-1]))
                    print("framework diversity: %d/%d" % (
                        len(set([utils._ml_framework_string(f)
                                 for f in valid_frameworks])),
                        len(valid_frameworks)))
                if i_episode > activate_h_controller:
                    print("n_hyperparams: %d" % n_hyperparams)
                print("")
        return self.tracker

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
                _ml_framework = clone(ml_framework)
                hyperparams, h_value_indices = \
                    self.select_hyperparameters(
                        t_state, component_probs, n_hyperparams, inner=True)
                _ml_framework = check_hyperparameters(
                    _ml_framework, self.a_space, hyperparams, h_value_indices)
                if _ml_framework is None:
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


def check_ml_framework(a_space, pipeline):
    """Check if the steps in ML framework form a valid pipeline."""
    # TODO: add more structure to an ml framework:
    # Data Preprocessor > Feature Preprocessor > Classifier
    try:
        assert hasattr(pipeline[-1].aclass, "predict")
        return a_space.create_ml_framework(pipeline, memory=None)
    except Exception:
        return None


def check_hyperparameters(
        ml_framework, a_space, hyperparameters, h_value_indices):
    """Check if the selected hyperp arameters are valid."""
    none_index = a_space.hyperparameter_state_space_keys.index("NONE_TOKEN")
    try:
        for h, i in h_value_indices.items():
            if i not in a_space.h_value_index(h) and i != none_index:
                return None
        return a_space.set_ml_framework_params(ml_framework, hyperparameters)
    except Exception:
        return None


def _performance_tracker():
    """Create a performance tracker tuple."""
    return utils.PerformanceTracker(
        best_candidates=[], best_scores=[], overall_mean_reward=[],
        overall_a_loss=[], overall_h_loss=[], overall_ml_performance=[],
        running_reward=10)


def _maintain_best_candidates(tracker, num_candidates, ml_framework, score):
    """Maintain the best candidates and their associated scores.

    NOTE: this function is stateful i.e. appends to best_candidates and
    best_scores list.
    """
    if len(tracker.best_candidates) < num_candidates:
        tracker.best_candidates.append(ml_framework)
        tracker.best_scores.append(score)
    else:
        min_index = tracker.best_scores.index(min(tracker.best_scores))
        if score > tracker.best_scores[min_index]:
            tracker.best_candidates[min_index] = ml_framework
            tracker.best_scores[min_index] = score
