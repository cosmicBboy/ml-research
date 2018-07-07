"""Module for generating algorithm and hyperparameter sequences.

This is a controller that jointly generates machine learning framework
pipelines and their associated hyperparameter settings. It does so via an RNN
that has a separate softmax classifier at each time step, where the classifier
has to predict either an algorithm to add to the pipeline or a hyperparameter
for the most recently chosen algorithm.

TODO:
- create an embedding layer for each algorithm/hyperparameter state space. This
  layer takes as input the action prediction and outputs a vector of size
  `embedding_size`. This should be the input_size.
  see https://github.com/titu1994/neural-architecture-search/blob/f9063db4e5d475f960e0aeb01600f5da78f13d0b/controller.py#L283  # noqa
- handle creation of an initial input vector.
  - option 1: create a randomly initialized vector that's the same
    dimensionality as the embedding (i.e. input) vector.
  - option 2: select a random one-hot encoded vector of the first state space,
    which would be fed into the RNN to get subsequent action predictions.
    This is how it's done here:
    https://github.com/titu1994/neural-architecture-search/blob/f9063db4e5d475f960e0aeb01600f5da78f13d0b/controller.py#L108
- implement get_action method, which takes as input previous action and outputs
  next action.
"""


import dill
import torch
import torch.nn as nn

from collections import defaultdict
from pathlib import Path
from torch.autograd import Variable
from torch.distributions import Categorical


class CASHController(nn.Module):
    """RNN module to generate joint algorithm and hyperparameter controller."""

    ALGORITHM = "algorithm"
    HYPERPARAMETER = "hyperparameter"

    def __init__(
            self, metafeature_size, input_size, hidden_size, output_size,
            a_space, optim=None, optim_kwargs=None, dropout_rate=0.1,
            num_rnn_layers=1):
        """Initialize Cash Controller.

        :param int metafeature_size:
        :param int input_size:
        :param int hidden_size:
        :param int output_size:
        :param AlgorithmSpace a_space:
        :param torch.optim optim:
        :param dict optim_kwargs:
        :param float dropout_rate:
        :param int num_rnn_layers:
        :ivar list[dict] action_classifiers:
        :ivar list[float] log_prob_buffer:
        :ivar list[float] reward buffer:
        """
        super(CASHController, self).__init__()
        self.a_space = a_space
        self.dropout_rate = dropout_rate
        self.num_rnn_layers = num_rnn_layers
        self.metafeature_size = metafeature_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.action_classifiers = []
        self.atype_map = {}
        self.algorithm_map = defaultdict(list)
        self.log_prob_buffer = []
        self.reward_buffer = []
        self.baseline_reward_buffer = []

        # architecture specification
        self.rnn = nn.GRU(
            self.input_size,
            self.hidden_size,
            num_layers=self.num_rnn_layers,
            dropout=self.dropout_rate)
        self.decoder = nn.Linear(self.hidden_size, self.output_size)
        self.dropout = nn.Dropout(self.dropout_rate)
        # special embedding layer for initial metafeature input
        self.metafeature_dense = nn.Linear(
            self.metafeature_size, self.input_size)

        # for each algorithm component and hyperparameter value, create a
        # softmax classifier over the number of unique components/hyperparam
        # values.
        # TODO: The architectue should be a branching structure that selects
        # the appropriate set of hyperparameter classifiers for a particular
        # choice of algorithm.
        self.component_dict = self.a_space.component_dict_from_signature()
        idx = 0
        for atype, components in self.component_dict.items():
            # action classifier
            self._add_action_classifier(idx, self.ALGORITHM, atype, components)
            self.atype_map[atype] = idx
            idx += 1
            for component in components:
                h_state_space = self.a_space.h_state_space([component])
                for hname, hvalues in h_state_space.items():
                    self._add_action_classifier(
                        idx, self.HYPERPARAMETER, hname, hvalues)
                    self.algorithm_map[component].append(idx)
                    idx += 1
        self.n_actions = len(self.action_classifiers)

        # set optimization object
        if optim is None:
            self.optim = optim
        else:
            self.optim = optim(self.parameters(), **optim_kwargs)

    def _add_action_classifier(self, action_index, action_type, name, choices):
        """Add action classifier to the nn.Module."""
        dense_attr = "action_dense_%s" % action_index
        softmax_attr = "action_softmax_%s" % action_index
        embedding_attr = "action_embedding_%s" % action_index
        # set layer attributes for action classifiers
        n_choices = len(choices)
        setattr(self, dense_attr, nn.Linear(self.output_size, n_choices))
        setattr(self, softmax_attr, nn.Softmax(1))
        setattr(self, embedding_attr, nn.Embedding(n_choices, self.input_size))
        # keep track of metadata for lookup purposes
        self.action_classifiers.append({
            "action_type": action_type,
            "name": name,
            "choices": choices,
            "dense_attr": dense_attr,
            "softmax_attr": softmax_attr,
            "embedding_attr": embedding_attr,
        })

    def forward(self, input, hidden, action_index):
        """Forward pass through controller."""
        if action_index == 0:
            # since the input of the first action is the metafeature vector,
            # transform metafeature vector into input dimensions
            input = self.metafeature_dense(input)
        rnn_output, hidden = self.rnn(input, hidden)
        rnn_output = self.dropout(self.decoder(rnn_output))

        # get appropriate action classifier for particular action_index
        action_classifier = self.action_classifiers[action_index]
        dense = getattr(self, action_classifier["dense_attr"])
        softmax = getattr(self, action_classifier["softmax_attr"])

        # obtain action probability distribution
        rnn_output = rnn_output.view(rnn_output.shape[0], -1)
        action_probs = softmax(dense(rnn_output))
        return action_probs, hidden

    def decode(self, init_input_tensor, init_hidden, init_index=0):
        """Decode a metafeature tensor to sequence of actions.

        Where the actions are a sequence of algorithm components and
        hyperparameter settings.
        """
        input_tensor, hidden = init_input_tensor, init_hidden
        actions = []
        # for each algorithm component type, select an algorithm
        for atype in self.component_dict:
            action, input_tensor, hidden = self._decode_action(
                input_tensor, hidden, self.atype_map[atype])
            actions.append(action)
            # each algorithm is associated with a set of hyperparameters
            for hyperparameter_index in self.algorithm_map[action["action"]]:
                action, input_tensor, hidden = self._decode_action(
                    input_tensor, hidden, hyperparameter_index)
                actions.append(action)
        return actions

    def _decode_action(self, input_tensor, hidden, action_index):
        action_probs, hidden = self.forward(
            input_tensor, hidden, action_index)
        action = self.select_action(action_probs, action_index)
        input_tensor = self.encode_embedding(
            action_index, action["choice_index"])
        return action, input_tensor, hidden

    def select_action(self, action_probs, action_index):
        """Select action based on action probability distribution."""
        # TODO: parameterize epsilon for eps-greedy implementation. At first
        # actions are sampled based on the action probability distribution,
        # but we want to eventually fully exploit the policy that we've learned
        # after N episodes.
        # Implement a epsilon `eps` schedule, which modulates the value of
        # `eps` from 0.5 (act greedy half of the time) to 1.0 (act greedy all
        # of the time).
        action_classifier = self.action_classifiers[action_index]
        action_dist = Categorical(action_probs)
        choice_index = action_dist.sample()
        _choice_index = int(action_dist.sample().data)
        return {
            "action_type": action_classifier["action_type"],
            "action_name": action_classifier["name"],
            "choices": action_classifier["choices"],
            "choice_index": _choice_index,
            "action": action_classifier["choices"][_choice_index],
            "action_log_prob": action_dist.log_prob(choice_index),
        }

    def encode_embedding(self, action_index, choice_index):
        """Encodes action choice into embedding at input for next action."""
        action_classifier = self.action_classifiers[action_index]
        embedding = getattr(self, action_classifier["embedding_attr"])
        action_embedding = embedding(
            Variable(torch.LongTensor([choice_index])))
        return action_embedding.view(
            1, action_embedding.shape[0], action_embedding.shape[1])

    def backward(self, show_grad=False, with_baseline=True):
        """End an episode with one backpropagation step.

        NOTE: This should be implemented at the ML framework controller level
        in order to implement the end-to-end RNN solution, where the
        algorithm and hyperparameter controller of the hidden layers are
        shared.

        # TODO: move this to the `cash_reinforce` module, since the learning
        # algorithm used to train the controller can be modular (right now
        # it's REINFORCE, but could be something else in the future).
        """
        if self.optim is None:
            raise ValueError(
                "optimization object not set. You need to provide `optim` "
                "and `optim_kwargs` when instantiating an %s object" % self)
        loss = []
        # compute loss
        for action_log_probs, reward, baseline_reward in zip(
                self.log_prob_buffer,
                self.reward_buffer,
                self.baseline_reward_buffer):
            for log_prob in action_log_probs:
                # since REINFORCE algorithm is a gradient ascent method,
                # negate the log probability in order to do gradient descent
                # on the negative expected rewards. If the reward is high and
                # the associated action log probabilities are close to 1, then
                # policy loss will be close to 0 (good). If reward is high and
                # the action log probs are close to 0, then policy loss will be
                # a large positive number (bad).
                r = reward - baseline_reward if with_baseline else reward
                loss.append(-log_prob * r)

        # one step of gradient descent
        self.optim.zero_grad()
        loss = torch.cat(loss).sum().div(len(self.reward_buffer))
        loss.backward()
        # gradient clipping to prevent exploding gradient
        nn.utils.clip_grad_norm(self.parameters(), 20)
        self.optim.step()

        if show_grad:
            print("\n\nGradients")
            for param in self.parameters():
                print(param.grad.data.sum())

        # reset rewards and log probs
        del self.reward_buffer[:]
        del self.log_prob_buffer[:]

        return loss.data[0]

    def init_hidden(self):
        return Variable(torch.zeros(self.num_rnn_layers, 1, self.hidden_size))

    def save(self, path):
        """Save weights and configuration."""
        path = Path(path)
        config_fp = path.parent / (path.stem + ".pkl")
        config = {
            "metafeature_size": self.metafeature_size,
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "a_space": self.a_space,
            "dropout_rate": self.dropout_rate,
            "num_rnn_layers": self.num_rnn_layers,
        }
        with config_fp.open("w+b") as fp:
            dill.dump(config, fp)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(self, path):
        """Load saved controller."""
        path = Path(path)
        config_fp = path.parent / (path.stem + ".pkl")
        with config_fp.open("r+b") as fp:
            rnn = self(**dill.load(fp))
        rnn.load_state_dict(torch.load(path))
        return rnn
