"""Module for generating algorithm and hyperparameter sequences.

This is a controller that jointly generates machine learning framework
pipelines and their associated hyperparameter settings. It does so via an RNN
that has a separate softmax classifier at each time step, where the classifier
has to predict either an algorithm to add to the pipeline or a hyperparameter
for the most recently chosen algorithm.
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
            self,
            metafeature_size,
            input_size,
            hidden_size,
            output_size,
            a_space,
            metafeature_encoding_size=20,
            aux_reward_size=1,
            dropout_rate=0.1,
            num_rnn_layers=1):
        """Initialize Cash Controller.

        :param int metafeature_size: dimensionality of metafeatures
        :param int input_size: dimensionality of input features (actions)
        :param int hidden_size: dimensionality of hidden RNN layer.
        :param int output_size: dimensionality of output space.
        :param AlgorithmSpace a_space:
        :param int metafeature_encoding_size: dim of metafeatures embedding
        :param float dropout_rate:
        :param int num_rnn_layers:
        :ivar int aux_size: dimensionality of auxiliary inputs (reward and
            action from previous time-step).
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

        # action dim for previous reward. Note that to implement meta-learning,
        # the previous action is encoded directly in the input.
        self.aux_reward_size = aux_reward_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.metafeature_encoding_size = metafeature_encoding_size

        # an array storing classifiers for algorithm components and
        # hyperparameter settings
        self.action_classifiers = []
        # maps algorithm types to index pointer referring to action classifier
        # for algorithm components of that type.
        self.atype_map = {}
        # maps algorithm component to list of index pointers referring to
        # action classifier for hyperparameters for that component
        self.acomponent_to_hyperparam = defaultdict(list)

        # buffers needed for backprop
        self.log_prob_buffer = []
        self.reward_buffer = []
        self.entropy_buffer = []

        # architecture specification
        self.rnn = nn.GRU(
            self.input_size + self.aux_reward_size +
            self.metafeature_encoding_size,
            self.hidden_size,
            num_layers=self.num_rnn_layers,
            dropout=self.dropout_rate)
        self.decoder = nn.Linear(self.hidden_size, self.output_size)
        self.dropout = nn.Dropout(self.dropout_rate)
        # special encoding layer for initial metafeature input
        self.metafeature_dense = nn.Linear(
            self.metafeature_size, self.metafeature_encoding_size)

        # for each algorithm component and hyperparameter value, create a
        # softmax classifier over the number of unique components/hyperparam
        # values.
        all_components = self.a_space.component_dict_from_signature(
            self.a_space.ALL_COMPONENTS)
        idx = 0
        for atype, components in all_components.items():
            # action classifier
            self._add_action_classifier(idx, self.ALGORITHM, atype, components)
            self.atype_map[atype] = idx
            idx += 1
            for component in components:
                h_state_space = self.a_space.h_state_space([component])
                for hname, hvalues in h_state_space.items():
                    self._add_action_classifier(
                        idx, self.HYPERPARAMETER, hname, hvalues)
                    self.acomponent_to_hyperparam[component].append(idx)
                    idx += 1
        self.n_actions = len(self.action_classifiers)

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

    def forward(self, input, aux, metafeatures, hidden, action_index):
        """Forward pass through controller."""
        input_concat = torch.cat([
            input, aux, self.metafeature_dense(metafeatures)], 2)
        rnn_output, hidden = self.rnn(input_concat, hidden)
        rnn_output = self.dropout(self.decoder(rnn_output))

        # get appropriate action classifier for particular action_index
        action_classifier = self.action_classifiers[action_index]
        dense = getattr(self, action_classifier["dense_attr"])
        softmax = getattr(self, action_classifier["softmax_attr"])

        # obtain action probability distribution
        rnn_output = rnn_output.view(rnn_output.shape[0], -1)
        action_probs = softmax(dense(rnn_output))
        return action_probs, hidden

    def decode(self, init_input_tensor, target_type, aux, metafeatures,
               init_hidden):
        """Decode a metafeature tensor to sequence of actions.

        Where the actions are a sequence of algorithm components and
        hyperparameter settings.

        TODO: add unit tests for this method and related methods.
        """
        input_tensor, hidden = init_input_tensor, init_hidden
        actions = []
        # for each algorithm component type, select an algorithm
        for atype in self.a_space.component_dict_from_target_type(
                target_type):
            action, input_tensor, hidden = self._decode_action(
                input_tensor, aux, metafeatures, hidden, self.atype_map[atype])
            actions.append(action)
            # each algorithm is associated with a set of hyperparameters
            for h_index in self.acomponent_to_hyperparam[action["action"]]:
                action, input_tensor, hidden = self._decode_action(
                    input_tensor, aux, metafeatures, hidden, h_index)
                actions.append(action)
        return actions, input_tensor

    def _decode_action(
            self, input_tensor, aux, metafeatures, hidden, action_index):
        action_probs, hidden = self.forward(
            input_tensor, aux, metafeatures, hidden, action_index)
        action = self.select_action(action_probs, action_index)
        input_tensor = self.encode_embedding(
            action_index, action["choice_index"])
        return action, input_tensor, hidden

    def select_action(self, action_probs, action_index):
        """Select action based on action probability distribution."""
        action_classifier = self.action_classifiers[action_index]
        log_action_probs = action_probs.log()
        choice_index = Categorical(action_probs).sample()
        _choice_index = int(choice_index.data)
        return {
            "action_type": action_classifier["action_type"],
            "action_name": action_classifier["name"],
            "choices": action_classifier["choices"],
            "choice_index": _choice_index,
            "action": action_classifier["choices"][_choice_index],
            "log_prob": log_action_probs.index_select(1, choice_index),
            "entropy": -(log_action_probs * action_probs).sum(1, keepdim=True)
        }

    def encode_embedding(self, action_index, choice_index):
        """Encode action choice into embedding at input for next action."""
        action_classifier = self.action_classifiers[action_index]
        embedding = getattr(self, action_classifier["embedding_attr"])
        action_embedding = embedding(
            Variable(torch.LongTensor([choice_index])))
        return action_embedding.view(
            1, action_embedding.shape[0], action_embedding.shape[1])

    def init_hidden(self):
        """Initialize hidden layer with zeros."""
        return Variable(torch.zeros(self.num_rnn_layers, 1, self.hidden_size))

    def init_action(self):
        """Initialize action input state."""
        return Variable(torch.zeros(1, 1, self.input_size))

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
    def load(cls, path):
        """Load saved controller."""
        path = Path(path)
        config_fp = path.parent / (path.stem + ".pkl")
        with config_fp.open("r+b") as fp:
            rnn = cls(**dill.load(fp))
        rnn.load_state_dict(torch.load(path))
        return rnn
