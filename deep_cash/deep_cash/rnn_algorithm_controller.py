"""Module for generating algorithms sequences.

TODO:
- remove dependency of MLFrameworkController on AlgorithmSpace
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


class _SubControllerRNN(nn.Module):
    """RNN module to generate algorithm component/hyperparameter settings.

    REINFORCE implementation adapted from:
    https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py
    """

    def __init__(
            self, metafeature_size, input_size, hidden_size, output_size,
            optim=None, optim_kwargs=None, dropout_rate=0.1, num_rnn_layers=1):
        """Initialize algorithm controller to propose ML frameworks."""
        super(_SubControllerRNN, self).__init__()
        self.dropout_rate = dropout_rate
        self.num_rnn_layers = num_rnn_layers
        self.metafeature_size = metafeature_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # architecture specification
        self.rnn = nn.GRU(
            metafeature_size + input_size, hidden_size,
            num_layers=self.num_rnn_layers)
        self.decoder = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.softmax = nn.Softmax(1)

        self.saved_log_probs = []
        self.rewards = []

        # set optimization object
        if optim is None:
            self.optim = optim
        else:
            self.optim = optim(self.parameters(), **optim_kwargs)

    def forward(self, metafeatures, input, hidden):
        input_concat = torch.cat((metafeatures, input), 2)
        output, hidden = self.rnn(input_concat, hidden)
        output = self.dropout(self.decoder(output))
        output = output.view(output.shape[0], -1)  # dim <seq_length x n_chars>
        output = self.softmax(output)
        return output, hidden

    def backward(
            self, baseline_reward, log_probs_attr="saved_log_probs",
            rewards_attr="rewards", show_grad=False):
        """End an episode with one backpropagation step.

        NOTE: This should be implemented at the ML framework controller level
        in order to implement the end-to-end RNN solution, where the
        algorithm and hyperparameter controller of the hidden layers are
        shared.
        """
        if self.optim is None:
            raise ValueError(
                "optimization object not set. You need to provide `optim` "
                "and `optim_kwargs` when instantiating an %s object" % self)
        loss = []
        log_probs = getattr(self, log_probs_attr)
        rewards = getattr(self, rewards_attr)

        # compute loss
        for log_prob, reward in zip(log_probs, rewards):
            loss.append(-log_prob * (reward - baseline_reward))

        # one step of gradient descent
        self.optim.zero_grad()
        loss = torch.cat(loss).sum().div(len(rewards))
        loss.backward()
        # gradient clipping to prevent exploding gradient
        nn.utils.clip_grad_norm(self.parameters(), 20)
        self.optim.step()

        if show_grad:
            print("\n\nGradients")
            for param in self.parameters():
                print(param.grad.data.sum())

        # reset rewards and log probs
        del rewards[:]
        del log_probs[:]

        return loss.data[0]

    def initHidden(self):
        return Variable(torch.zeros(self.num_rnn_layers, 1, self.hidden_size))


class AlgorithmControllerRNN(_SubControllerRNN):
    """RNN module to generate algorithm components."""
    pass


class HyperparameterControllerRNN(_SubControllerRNN):
    """RNN module to propose hyperparameter settings."""

    def __init__(self, *args, **kwargs):
        super(HyperparameterControllerRNN, self).__init__(*args, **kwargs)
        # lists to store inner hyperparameter training loop.
        self.inner_saved_log_probs = []
        self.inner_rewards = []

    def inner_backward(self, baseline_reward, **kwargs):
        self.backward(
            baseline_reward, log_probs_attr="inner_saved_log_probs",
            rewards_attr="inner_rewards", **kwargs)


class MLFrameworkController(object):
    """Controller to generate machine learning frameworks."""

    def __init__(self, a_controller, h_controller, a_space):
        self.a_controller = a_controller
        self.h_controller = h_controller
        self.a_space = a_space

    def select_ml_framework(self, t_state):
        """Select ML framework from the algorithm controller given task state.

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
        hidden = self.a_controller.initHidden()
        ml_framework = []
        # TODO: eventually remove this so that the hyperparameter controller
        # doesn't need the flattened softmax output of the algorithm controller
        component_probs = []
        # compute joint log probability of the components
        log_probs = []
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
        hyperparameter_tensor = utils._create_hyperparameter_tensor(
            self.a_space, [self.a_space.h_start_token_index])
        h_hidden = self.h_controller.initHidden()
        h_log_probs = []
        hyperparameters = []
        h_value_indices = []
        # compute joint log probability of hyperparameters
        for h_key in self.a_space.hyperparameter_name_space[:n_hyperparams]:
            h_probs, h_hidden = self.h_controller(
                metafeature_tensor, hyperparameter_tensor, h_hidden)
            h_m = Categorical(h_probs)
            h_action = h_m.sample()
            h_log_probs.append(h_m.log_prob(h_action))
            h_value = \
                self.a_space.hyperparameter_state_space_values[int(h_action)]
            h_value_indices.append((h_key, int(h_action)))
            hyperparameters.append((h_key, h_value))
            hyperparameter_tensor = utils._create_hyperparameter_tensor(
                self.a_space, [int(h_action)])
        if inner:
            self.h_controller.inner_saved_log_probs.append(
                torch.cat(h_log_probs).sum())
        else:
            self.h_controller.saved_log_probs.append(
                torch.cat(h_log_probs).sum())
        return OrderedDict(hyperparameters), OrderedDict(h_value_indices)

    def fit(self):
        pass

    def fit_h_controller(self):
        pass


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


def train_h_controller(
        mlf_controller, t_env, t_state, component_probs,
        ml_framework, n_iter, verbose=False, n_hyperparams=1):
    """Train the hyperparameter controller given an ML framework."""
    # TODO: need to create a separate inner loop for training the
    # the h_controller. For each valid ml_framework, perform
    # an inner loop with h_controller's own reward that signals
    # whether or not a particular hyperparameter proposal is valid
    # given the ml_framework.
    h_current_baseline_reward = 0
    h_prev_baseline_reward = 0
    if verbose:
        print("\n%s" % utils._ml_framework_string(ml_framework))
    for h_i_episode in range(n_iter):
        n_valid_hyperparams = 0
        for i in range(100):
            tmp_ml_framework = clone(ml_framework)
            hyperparameters, h_value_indices = \
                mlf_controller.select_hyperparameters(
                    t_state, component_probs, n_hyperparams, inner=True)
            tmp_ml_framework = check_hyperparameters(
                tmp_ml_framework, mlf_controller.a_space, hyperparameters,
                h_value_indices)
            if tmp_ml_framework is None:
                reward = t_env.error_reward
            else:
                n_valid_hyperparams += 1
                reward = t_env.correct_hyperparameter_reward
            mlf_controller.h_controller.inner_rewards.append(reward)
            h_current_baseline_reward = utils._exponential_mean(
                reward, h_current_baseline_reward)
            if verbose:
                print("Ep%d, %d/%d valid hyperparams %s%s" %
                      (h_i_episode, n_valid_hyperparams, i + 1,
                       h_value_indices, " " * 10),
                      sep=" ", end="\r", flush=True)
        h_loss = mlf_controller.h_controller.inner_backward(
            h_prev_baseline_reward)
        h_prev_baseline_reward = h_current_baseline_reward
    return h_loss, h_current_baseline_reward


def _performance_tracker():
    return {
        "best_candidates": [],
        "best_scores": [],
        "overall_mean_reward": [],
        "overall_a_loss": [],
        "overall_h_loss": [],
        "overall_ml_performance": [],
        "running_reward": 10,
    }


def _maintain_best_candidates(tracker, num_candidates, ml_framework, score):
    """Maintain the best candidates and their associated scores.

    NOTE: this function is stateful i.e. appends to best_candidates and
    best_scores list.
    """
    if len(tracker["best_candidates"]) < num_candidates:
        tracker["best_candidates"].append(ml_framework)
        tracker["best_scores"].append(score)
    else:
        min_index = tracker["best_scores"].index(min(tracker["best_scores"]))
        if score > tracker["best_scores"][min_index]:
            tracker["best_candidates"][min_index] = ml_framework
            tracker["best_scores"][min_index] = score


def train(a_controller, h_controller, a_space, t_env,
          num_episodes=10, log_every=1, n_iter=1000, show_grad=False,
          num_candidates=10, activate_h_controller=2,
          increase_n_hyperparam_by=3, increase_n_hyperparam_every=5):
    """Train the AlgorithmContoller RNN.

    :param AlgorithmControllerRNN a_controller: Controller to sample
        transformers/estimators from the algorithm space.
    :param AlgorithmSpace a_space: An object containing possible algorithm
        space.
    :param DataSpace t_env: An environment that generates tasks to do on a
        particular dataset and evaluates the solution proposed by the
        algorithm controller.
    :param int increase_n_hyperparam_by: increase number of hyperparameters
        by this much.
    :param int increase_n_hyperparam_every: increase number of hyperparameters
        to propose after this many episodes.
    """
    mlf_controller = MLFrameworkController(a_controller, h_controller, a_space)
    tracker = _performance_tracker()
    prev_baseline_reward = 0
    last_valid_framework = None
    n_hyperparams = 1  # propose this many hyperparameters
    for i_episode in range(num_episodes):
        # sample data environment from data distribution
        t_env.sample_data_env()
        # sample training/test data from data environment
        t_state = t_env.sample()
        n_valid_frameworks = 0
        n_valid_hyperparams = 0
        n_successful = 0
        current_baseline_reward = 0
        ml_performance = []
        valid_frameworks = []

        # increment number of hyperparameters to predict
        if i_episode > 0 and i_episode > activate_h_controller and \
                i_episode % increase_n_hyperparam_every == 0:
            n_hyperparams += increase_n_hyperparam_by

        for i in range(n_iter):
            # ml framework pipeline creation
            pipeline, component_probs = \
                mlf_controller.select_ml_framework(t_state)
            ml_framework = check_ml_framework(a_space, pipeline)
            if ml_framework is None:
                reward = t_env.error_reward
            else:
                n_valid_frameworks += 1

                # hyperparameter controller inner and outer loop
                if i_episode > activate_h_controller:
                    _, _ = train_h_controller(
                        mlf_controller, t_env, t_state, component_probs,
                        clone(ml_framework), n_iter=1,
                        n_hyperparams=n_hyperparams)
                    hyperparameters, h_value_indices = \
                        mlf_controller.select_hyperparameters(
                            t_state, component_probs,
                            n_hyperparams=n_hyperparams)
                    ml_framework = check_hyperparameters(
                        ml_framework, a_space, hyperparameters,
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
                        last_valid_framework = ml_framework
                        valid_frameworks.append(ml_framework)
                        n_successful += 1
                        ml_performance.append(reward)
                        _maintain_best_candidates(
                            tracker, num_candidates, ml_framework, reward)
            print("%d/%d valid frameworks, %d/%d valid hyperparams "
                  "%d/%d successful frameworks" %
                  (n_valid_frameworks, i + 1,
                   n_valid_hyperparams, i + 1,
                   n_successful, i + 1),
                  sep=" ", end="\r", flush=True)
            current_baseline_reward = utils._exponential_mean(
                reward, current_baseline_reward)
            t_state = t_env.sample()
            mlf_controller.a_controller.rewards.append(reward)
            mlf_controller.h_controller.rewards.append(reward)

        tracker["running_reward"] = utils._exponential_mean(
            tracker["running_reward"], i)
        tracker["overall_mean_reward"].append(
            np.mean(mlf_controller.a_controller.rewards))
        tracker["overall_ml_performance"].append(
            np.mean(ml_performance) if len(ml_performance) > 0 else np.nan)

        if i_episode > activate_h_controller:
            h_loss = mlf_controller.h_controller.backward(prev_baseline_reward)
            tracker["overall_h_loss"].append(h_loss)
        else:
            tracker["overall_h_loss"].append(np.nan)

        # backward pass
        a_loss = mlf_controller.a_controller.backward(prev_baseline_reward)
        tracker["overall_a_loss"].append(a_loss)
        # update baseline rewards
        prev_baseline_reward = current_baseline_reward
        if i_episode % log_every == 0:
            print("\nEp%s | mean reward: %0.02f | "
                  "mean perf: %0.02f | ep length: %d | "
                  "running reward: %0.02f" %
                  (i_episode, tracker["overall_mean_reward"][i_episode],
                   tracker["overall_ml_performance"][i_episode], i + 1,
                   tracker["running_reward"]))
            if last_valid_framework:
                print("last framework: %s" %
                      utils._ml_framework_string(last_valid_framework))
            if len(valid_frameworks) > 0:
                print("framework diversity: %d/%d" % (
                    len(set([utils._ml_framework_string(f)
                             for f in valid_frameworks])),
                    len(valid_frameworks)))
            if i_episode > activate_h_controller:
                print("n_hyperparams: %d" % n_hyperparams)
            print("")

    return tracker
