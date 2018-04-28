"""Module for generating algorithms sequences."""

import numpy as np
import torch
import torch.nn as nn

from torch.autograd import Variable
from torch.distributions import Categorical

from .algorithm_space import START_TOKEN

EPS = np.finfo(np.float32).eps.item()
GAMMA = 0.99


# specify type of the metafeature
METAFEATURES = [
    ("number_of_examples", int)
]


class AlgorithmControllerRNN(nn.Module):

    """RNN module to generate algorithm components.

    REINFORCE implementation adapted from:
    https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py
    """

    def __init__(
            self, metafeature_size, input_size, hidden_size, output_size,
            dropout_rate=0.1, num_rnn_layers=1):
        super(AlgorithmControllerRNN, self).__init__()
        self.dropout_rate = dropout_rate
        self.num_rnn_layers = num_rnn_layers
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.metafeature_size = metafeature_size
        self.rnn = nn.GRU(
            metafeature_size + input_size, hidden_size,
            num_layers=self.num_rnn_layers)
        self.decoder = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.softmax = nn.Softmax(1)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, metafeatures, input, hidden):
        input_concat = torch.cat((metafeatures, input), 2)
        output, hidden = self.rnn(input_concat, hidden)
        output = self.dropout(self.decoder(output))
        output = output.view(output.shape[0], -1)  # dim <seq_length x n_chars>
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        # randomly initialize
        return Variable(torch.randn(self.num_rnn_layers, 1, self.hidden_size))


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
    """Convert a string of characters to an input tensor.

    Returns tensor of dim <string_length x 1 x n_characters>."""
    t = torch.zeros(len(seq), 1, a_space.n_components)
    for i, action in enumerate(seq):
        t[i][0][a_space.components.index(action)] = 1
    return t


def _create_training_data_tensors(a_space, metafeatures, seq):
    return (
        Variable(_create_metafeature_tensor(metafeatures, seq)),
        Variable(_create_input_tensor(a_space, seq)))


def select_ml_framework(a_controller, a_space, metafeatures):
    """Select ML framework from the algrothm controller given state.

    The task state is implemented as the metafeatures associated with a
    particular training set metafeatures, in the context of bootstrap sampling
    where an ML framework is fit on the bootstrapped sample and the validation
    performance (reward) is computed on the out-of-bag sample.

    :param AlgorithmControllerRNN a_controller:
    :param list metafeatures:
    """
    # TODO:
    # - create metafeatures tensor, which has dim <1 x 1 x n_metafeatures>
    # - create input tensor, which has dim <1 x 1 x a_space.n_components>,
    #   where the 3rd dimension encodes a one-hot encoded vector the index
    #   for the <sos> token is 1.
    # - Note that the first dimension of the metafeatures and input tensor are
    #   both 1 because we want to train a generative model that proposes
    #   policies, which are sequences of actions, each action specifying one
    #   aspect of a machine learning framework
    metafeature_tensor, input_tensor = _create_training_data_tensors(
        a_space, metafeatures, [START_TOKEN])
    hidden = a_controller.initHidden()
    ml_framework = []

    # compute joint log probability of the components
    log_probs = []
    for i in range(a_space.N_COMPONENT_TYPES):
        probs, hidden = a_controller(metafeature_tensor, input_tensor, hidden)
        m = Categorical(probs[0])
        action = m.sample()
        component = a_space.components[action.data[0]]
        ml_framework += [component]
        input_tensor = Variable(_create_input_tensor(a_space, [component]))
        log_probs.append(m.log_prob(action))
    log_probs = torch.cat(log_probs)
    a_controller.saved_log_probs.append(log_probs.sum())
    return ml_framework


def terminate_episode(a_controller, optim):
    R = 0
    policy_loss = []
    rewards = []

    # compute discounted reward
    for r in a_controller.rewards[::-1]:
        R = r + GAMMA * R
        rewards.insert(0, R)
    rewards = torch.Tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + EPS)

    # compute loss
    for log_prob, reward in zip(a_controller.saved_log_probs, rewards):
        policy_loss.append(-log_prob * reward)

    # one step of gradient descent
    optim.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optim.step()

    # reset rewards and log probs
    del a_controller.rewards[:]
    del a_controller.saved_log_probs[:]

    return policy_loss.data[0]


def check_ml_framework(a_space, ml_framework):
    try:
        return a_space.create_ml_framework(ml_framework)
    except:
        return None


def train(a_controller, a_space, t_env, optim, num_episodes=10,
          log_every=1, n_iter=5000):
    """Train the AlgorithmContoller RNN.

    :param AlgorithmControllerRNN a_controller: Controller to sample
        transformers/estimators from the algorithm space.
    :param AlgorithmSpace a_space: An object containing possible algorithm
        space.
    :param DataSpace t_env: An environment that generates tasks to do on a
        particular dataset and evaluates the solution proposed by the
        algorithm controller.
    """
    running_reward = 10
    for i_episode in range(num_episodes):
        t_state = t_env.sample()  # initial sample
        n_valid = 0
        ml_score = []
        for i in range(n_iter):
            ml_framework = select_ml_framework(a_controller, a_space, t_state)
            ml_framework = check_ml_framework(a_space, ml_framework)
            if ml_framework is None:
                reward = -1  # negative reward for proposing invalid framework
            else:
                n_valid += 1
                print("Proposed %d/%d valid frameworks" % (n_valid, i + 1),
                      sep=" ", end="\r", flush=True)
                reward = t_env.evaluate(ml_framework)
                ml_score.append(reward)
            t_state = t_env.sample()
            a_controller.rewards.append(reward)

        running_reward = running_reward * 0.99 + i * 0.01
        loss = terminate_episode(a_controller, optim)
        if i_episode % log_every == 0:
            print("\nEpisode: %s | Loss: %0.02f | Mean Score: %0.02f | "
                  "Last episode length: %d | Running reward: %0.02f" %
                  (i_episode, loss, np.mean(ml_score), i + 1, running_reward))

