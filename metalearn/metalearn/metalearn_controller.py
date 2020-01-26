"""Module for generating algorithm and hyperparameter sequences.

This is a controller that jointly generates machine learning framework
pipelines and their associated hyperparameter settings. It does so via an RNN
that has a separate softmax classifier at each time step, where the classifier
has to predict either an algorithm to add to the pipeline or a hyperparameter
for the most recently chosen algorithm.
"""

from collections import defaultdict

import dill
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical

from .data_types import CASHComponent, TargetType
from .components.algorithm_component import EXCLUDE_ALL


logger = logging.getLogger(__name__)


class MetaLearnController(nn.Module):
    """RNN module to generate joint algorithm and hyperparameter controller."""

    def __init__(
            self,
            metafeature_size,
            input_size,
            hidden_size,
            output_size,
            a_space,
            mlf_signature=None,
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
        :param list[str] mlf_signature:
        :param int metafeature_encoding_size: dim of metafeatures embedding
        :param float dropout_rate:
        :param int num_rnn_layers:
        :ivar int aux_size: dimensionality of auxiliary inputs (reward and
            action from previous time-step).
        :ivar list[dict] action_classifiers:
        :ivar list[float] log_prob_buffer:
        :ivar list[float] reward buffer:
        """
        super(MetaLearnController, self).__init__()
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
        self.acomponent_to_hyperparams = defaultdict(list)
        # maps hyperparameter actions to conditional masks that determines what
        # subsequent actions the agent can make.
        self.hyperparam_exclude_conditions = defaultdict(dict)

        # buffers needed for backprop
        self.value_buffer = []
        self.log_prob_buffer = []
        self.reward_buffer = []
        self.entropy_buffer = []

        # architecture specification

        # meta-rnn models representation of previous action, previous reward,
        # and metafeatures
        self.meta_rnn = nn.GRU(
            self.input_size + self.aux_reward_size +
            self.metafeature_encoding_size,
            self.hidden_size,
            num_layers=self.num_rnn_layers,
            dropout=self.dropout_rate,
        )
        self.meta_decoder = nn.Linear(self.hidden_size, self.output_size)
        self.meta_dropout = nn.Dropout(self.dropout_rate)

        # special encoding layer for initial metafeature input
        self.metafeature_dense = nn.Linear(
            self.metafeature_size, self.metafeature_encoding_size)

        # critic layer that produces value estimate
        self.critic_dropout = nn.Dropout(self.dropout_rate)
        self.critic_linear1 = nn.Linear(self.output_size, self.hidden_size)
        self.critic_linear2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.critic_linear3 = nn.Linear(self.hidden_size, self.hidden_size)
        self.critic_output = nn.Linear(self.hidden_size, 1)

        # micro action policy rnn takes meta-rnn representation and selects
        # hyperparameters to specify an ML Framework.
        self.micro_action_rnn = nn.GRU(
            self.input_size,
            self.hidden_size,
            num_layers=self.num_rnn_layers,
            dropout=self.dropout_rate,
        )
        self.micro_action_decoder = nn.Linear(
            self.hidden_size, self.output_size)
        self.micro_action_dropout = nn.Dropout(self.dropout_rate)

        # for each algorithm component and hyperparameter value, create a
        # softmax classifier over the number of unique components/hyperparam
        # values.
        self._mlf_signature = mlf_signature
        all_components = self.a_space.component_dict_from_signature(
            self.a_space.ALL_COMPONENTS if self._mlf_signature is None
            else self._mlf_signature)
        idx = 0

        for atype, acomponents in all_components.items():
            # action classifier
            self._add_action_classifier(
                idx, CASHComponent.ALGORITHM, atype, acomponents, None)
            self.atype_map[atype] = idx
            idx += 1
            for acomponent in acomponents:
                h_state_space = self.a_space.h_state_space([acomponent])
                h_exclude_cond_map = self.a_space.h_exclusion_conditions(
                    [acomponent])
                for hname, hchoices in h_state_space.items():
                    exclude_masks = self._exclusion_condition_to_mask(
                        h_state_space, h_exclude_cond_map.get(hname, None))
                    self._add_action_classifier(
                        idx, CASHComponent.HYPERPARAMETER, hname, hchoices,
                        exclude_masks)
                    self.acomponent_to_hyperparams[acomponent].append(idx)
                    idx += 1
        self.n_actions = len(self.action_classifiers)

    def _exclusion_condition_to_mask(self, h_state_space, exclude):
        """
        Create a mask of the action space to prevent selecting invalid states
        given previous choices. A mask value 1 means to exclude the value from
        the state space.
        """

        if exclude is None:
            return None

        exclude_masks = defaultdict(dict)
        for choice, exclude_conditions in exclude.items():
            for hname, exclude_values in exclude_conditions.items():
                if exclude_values == EXCLUDE_ALL:
                    mask = [1 for _ in h_state_space[hname]]
                else:
                    mask = [
                        int(i in exclude_values) for i in h_state_space[hname]]
                exclude_masks[choice].update({hname: torch.ByteTensor(mask)})

        return exclude_masks

    def _add_action_classifier(
            self, action_index, action_type, name, choices,
            exclude_masks):
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
            "exclude_masks": exclude_masks,
        })

    def get_policy_dist(self, action, hidden, action_classifier, mask=None):
        """Get action distribution from policy."""
        rnn_output, hidden = self.micro_action_rnn(action, hidden)
        rnn_output = self.micro_action_dropout(
            self.micro_action_decoder(rnn_output))

        dense = getattr(self, action_classifier["dense_attr"])
        softmax = getattr(self, action_classifier["softmax_attr"])

        # obtain action probability distribution
        rnn_output = rnn_output.view(rnn_output.shape[0], -1)
        logits = dense(rnn_output)
        if mask is not None:
            logger.warning(
                "applying mask %s to action_classifier %s" %
                (mask, action_classifier))
            logits[mask] = float("-inf")
        action_probs = softmax(logits)
        return action_probs, hidden

    def get_value(self, state):
        """Get value estimate from state.

        The state includes dataset metafeatures, previous action, and previous
        reward.
        """
        value = self.critic_dropout(F.relu(self.critic_linear1(state)))
        value = self.critic_dropout(F.relu(self.critic_linear2(value)))
        value = self.critic_dropout(F.relu(self.critic_linear3(value)))
        return self.critic_output(value)

    def forward(
            self,
            prev_action: torch.FloatTensor,
            prev_reward: torch.FloatTensor,
            metafeatures: torch.FloatTensor,
            hidden: torch.FloatTensor,
            target_type: TargetType):
        """Decode a metafeature tensor to sequence of actions.

        Where the actions are a sequence of algorithm components and
        hyperparameter settings.

        :prev_action: tensor of action distribution at previous time step.
        :prev_reward: single-element tensor of reward at previous time step.
        :metafeatures: tensor of dataset feature representation.
        :hidden: hidden state of previous time step.
        :target_type: type of supervision target
        """
        actions = []

        # first process inputs with metalearning rnn
        state, hidden = self.meta_rnn(
            torch.cat(
                [
                    prev_action,
                    prev_reward,
                    F.relu(self.metafeature_dense(metafeatures))
                ],
                dim=2
            ),
            hidden)
        action_tensor = state = self.meta_dropout(
            self.meta_decoder(state))

        # get value estimate
        value = self.get_value(state)

        # create attribute dictionary that maps hyperparameter names to
        # conditional masks
        self._exclude_masks = {}
        # for each algorithm component type, select an algorithm
        if self._mlf_signature is None:
            algorithm_components = \
                self.a_space.component_dict_from_target_type(target_type)
        else:
            algorithm_components = \
                self.a_space.component_dict_from_signature(self._mlf_signature)

        for atype in algorithm_components:
            action, action_tensor, hidden = self._decode_action(
                action_tensor, hidden, action_index=self.atype_map[atype])
            if action is None:
                raise RuntimeError(
                    "selected action for algorithm component cannot be None.\n"
                    "exclude masks: %s\n"
                    "algorithm components: %s\n"
                    "algorithm type: %s" % (
                        self._exclude_masks,
                        algorithm_components,
                        atype,
                    )
                )
            actions.append(action)
            # each algorithm is associated with a set of hyperparameters
            for h_index in self.acomponent_to_hyperparams[action["choice"]]:
                hyperparameter_action, action_tensor, hidden = \
                    self._decode_action(
                        action_tensor, hidden, action_index=h_index)
                if hyperparameter_action is None:
                    logger.warning(
                        "selected action for hyperparameter index %s for "
                        "algorithm choice %s" % (h_index, action["choice"])
                    )
                    continue
                actions.append(hyperparameter_action)
        return value, actions, action_tensor, hidden

    def _get_exclusion_mask(self, action_name):
        if action_name in self._exclude_masks:
            return (
                self._exclude_masks[action_name]
                .view(1, -1).type(torch.bool)
            )
        else:
            return None

    def _accumulate_exclusion_mask(self, masks):
        for action_name, m in masks.items():
            mask = self._exclude_masks.get(action_name, None)
            if mask is None:
                self._exclude_masks[action_name] = m
            else:
                acc_mask = (m.int() + mask.int()) > 0
                acc_mask = acc_mask.type(torch.ByteTensor)
                self._exclude_masks[action_name] = acc_mask

    def _decode_action(
            self, action_tensor, hidden, action_index):
        action_classifier = self.action_classifiers[action_index]
        mask = self._get_exclusion_mask(action_classifier["name"])
        if mask is not None and (mask == 1).all():
            # If all actions are masked, then don't select any choice
            logger.warning(
                "all actions are masked for action_classifier %s" %
                action_classifier)
            return None, action_tensor, hidden
        action_probs, hidden = self.get_policy_dist(
            action_tensor, hidden, action_classifier, mask)
        action = self._select_action(action_probs, action_classifier, mask)
        if action is None:
            return None, action_tensor, hidden
        action_tensor = self.encode_embedding(
            action_index, action["choice_index"])
        return action, action_tensor, hidden

    def _select_action(self, action_probs, action_classifier, mask):
        """Select action based on action probability distribution."""
        # compute unmasked log probabilitises.
        policy_dist = Categorical(action_probs)
        choice_index = policy_dist.sample()
        _choice_index = choice_index.data.item()
        choice = action_classifier["choices"][_choice_index]

        # for current choice, accumulate exclusion masks if any
        if action_classifier["exclude_masks"] is not None and \
                choice in action_classifier["exclude_masks"]:
            for _, masks in action_classifier["exclude_masks"].items():
                self._accumulate_exclusion_mask(masks)

        return {
            "action_type": action_classifier["action_type"],
            "action_name": action_classifier["name"],
            "choices": action_classifier["choices"],
            "choice_index": _choice_index,
            "choice": choice,
            "log_prob": policy_dist.log_prob(choice_index),
            "entropy": policy_dist.entropy(),
        }

    def encode_embedding(self, action_index, choice_index):
        """Encode action choice into embedding at input for next action."""
        action_classifier = self.action_classifiers[action_index]
        embedding = getattr(self, action_classifier["embedding_attr"])
        action_embedding = embedding(torch.LongTensor([choice_index]))
        return action_embedding.view(
            1, action_embedding.shape[0], action_embedding.shape[1])

    def init_hidden(self):
        """Initialize hidden layer with zeros."""
        return torch.zeros(
            self.num_rnn_layers, 1, self.hidden_size, requires_grad=True)

    def init_action(self):
        """Initialize action input state."""
        return torch.zeros(1, 1, self.input_size, requires_grad=True)

    @property
    def config(self):
        return {
            "metafeature_size": self.metafeature_size,
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "a_space": self.a_space,
            "dropout_rate": self.dropout_rate,
            "num_rnn_layers": self.num_rnn_layers,
        }

    def save(self, path):
        """Save weights and configuration."""
        torch.save(
            {
                "config": self.config,
                "weights": self.state_dict(),
            },
            path,
            pickle_module=dill)

    @classmethod
    def load(cls, path):
        """Load saved controller."""
        model_config = torch.load(path, pickle_module=dill)
        rnn = cls(**model_config["config"])
        rnn.load_state_dict(model_config["weights"])
        return rnn
