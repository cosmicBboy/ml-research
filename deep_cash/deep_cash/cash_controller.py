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


import torch
import torch.nn as nn

from torch.autograd import Variable


class CASHController(nn.Module):
    """RNN module to generate joint algorithm and hyperparameter controller."""

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

        # architecture specification
        self.rnn = nn.GRU(
            self.metafeature_size + self.input_size, self.hidden_size,
            num_layers=self.num_rnn_layers, dropout=self.dropout_rate)
        self.decoder = nn.Linear(self.hidden_size, self.output_size)
        self.dropout = nn.Dropout(self.dropout_rate)

        setattr(self, "softmax", nn.Softmax(1))

        # for each algorithm component and hyperparameter value, create a
        # softmax classifier over the number of unique components/hyperparam
        # values.
        # NOTE: What if the action space was such that the controller proposes
        # all the algorithms first and then proposes the hyperparameters?
        self.component_dict = self.a_space.component_dict_from_signature()
        idx = 0
        for atype, components in self.component_dict.items():
            # action classifier
            self._add_action_classifier(idx, "algorithm", atype, components)
            idx += 1
            # all hyperparameter state spaces include the special token
            # "<none>", which effectively turns it off.
            h_state_space = self.a_space.h_state_space(
                components, with_none_token=True)
            for hname, hvalues in h_state_space.items():
                self._add_action_classifier(
                    idx, "hyperparameter", hname, hvalues)
                idx += 1

        # set optimization object
        if optim is None:
            self.optim = optim
        else:
            self.optim = optim(self.parameters(), **optim_kwargs)

    def _add_action_classifier(self, action_index, action_type, name, choices):
        """Adds action classifier to the nn.Module."""
        dense_attr = "action_dense_%s" % action_index
        softmax_attr = "action_softmax_%s" % action_index
        # set layer attributes
        setattr(self, dense_attr, nn.Linear(self.output_size, len(choices)))
        setattr(self, softmax_attr, nn.Softmax(1))
        # keep track of metadata for lookup purposes
        self.action_classifiers.append({
            "action_type": action_type,
            "name": name,
            "choices": choices,
            "dense_attr": dense_attr,
            "softmax_attr": softmax_attr,
        })

    def forward(self, metafeatures, input, hidden, action_index):
        input_concat = torch.cat((metafeatures, input), 2)
        output, hidden = self.rnn(input_concat, hidden)
        output = self.dropout(self.decoder(output))

        # get appropriate action classifier for particular action_index
        action_classifier = self.action_classifiers[action_index]
        dense = getattr(self, action_classifier["dense_attr"])
        softmax = getattr(self, action_classifier["softmax_attr"])

        # obtain action
        output_flat = output.view(output.shape[0], -1)
        action = softmax(dense(output_flat))
        return output, hidden, action

    def backward(self):
        pass

    def init_hidden(self):
        return Variable(torch.zeros(self.num_rnn_layers, 1, self.hidden_size))
