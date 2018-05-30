"""Module for generating algorithms sequences.

TODO:
- remove dependency of MLFrameworkController on AlgorithmSpace
- add time cost to fitting a proposed framework
"""

import torch
import torch.nn as nn

from torch.autograd import Variable


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
            rewards_attr="rewards", show_grad=False,
            baseline_aslist=False):
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
        for i, (log_prob, reward) in enumerate(zip(log_probs, rewards)):
            b = baseline_reward[i] if baseline_aslist else baseline_reward
            loss.append(-log_prob * (reward - b))

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

    def init_hidden(self):
        return Variable(torch.zeros(self.num_rnn_layers, 1, self.hidden_size))


class AlgorithmControllerRNN(_SubControllerRNN):
    """RNN module to generate algorithm components."""
    pass


class HyperparameterControllerRNN(_SubControllerRNN):
    """RNN module to propose hyperparameter settings."""

    def __init__(self, *args, **kwargs):
        """Initialize hyperparameter controller."""
        super(HyperparameterControllerRNN, self).__init__(*args, **kwargs)
        # lists to store inner hyperparameter training loop.
        self.inner_saved_log_probs = []
        self.inner_rewards = []

    def inner_backward(self, baseline_reward, **kwargs):
        """Backward pass through the inner loop.

        The inner training loop enables the hyperparameter controller to
        adjust its weights by proposing hyperparameters conditioned on a single
        ml pipeline.
        """
        self.backward(
            baseline_reward, log_probs_attr="inner_saved_log_probs",
            rewards_attr="inner_rewards", baseline_aslist=True, **kwargs)
