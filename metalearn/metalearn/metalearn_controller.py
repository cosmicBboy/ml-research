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

from torch.distributions import Categorical, Normal

from .data_types import CASHComponent, TargetType, HyperparamType
from .components.algorithm_component import EXCLUDE_ALL


logger = logging.getLogger(__name__)


CONTINUOUS_HYPERPARAM_TYPES = {HyperparamType.INTEGER, HyperparamType.REAL}


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
        num_rnn_layers=1,
        action_meta=None,
        action_index=None,
    ):
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
        :param action_meta: dictionary of action metadata for pre-trained
            controller
        :param action_index: dictionary of action pointers for pre-trained
            controller
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
        self.mlf_signature = mlf_signature
          # map action keys to nn.Modules
        self.action_modules = nn.ModuleDict()
         # map action keys to action metadata
        self.action_meta = {} if action_meta is None else action_meta
        self.action_index = {
            # map algorithm type to algorithm components
            "algo_type": {},
            # map algorithm component to list of keys of its hyperparameters
            "algo_to_hp": defaultdict(list),
        }

        all_components = self.a_space.component_dict_from_signature(
            self.a_space.ALL_COMPONENTS if self.mlf_signature is None
            else self.mlf_signature)

        # TODO: make sure values of `all_components` can't be empty lists
        for atype, acomponents in all_components.items():
            # action classifier
            algo_type_key = f"algo_{atype.value}"
            self.action_modules[algo_type_key] = nn.ModuleDict({
                "dense": nn.Linear(self.output_size, len(acomponents)),
                "softmax": nn.Softmax(1),
                "embedding": nn.Linear(len(acomponents), self.input_size),
            })
            self.action_meta[algo_type_key] = {
                "action_type": CASHComponent.ALGORITHM,
                "name": atype.value,
                "choices": acomponents,
                "exclude_masks": None,
            }
            self.action_index["algo_type"][atype] = algo_type_key

            for acomponent in acomponents:
                hp_state_space = self.a_space.h_state_space([acomponent])
                hp_exclude_cond_map = self.a_space.h_exclusion_conditions(
                    [acomponent]
                )
                for hname, h_meta in hp_state_space.items():
                    if h_meta["type"] == HyperparamType.CATEGORICAL:
                        action_head_layers = {
                            "dense": nn.Linear(
                                self.output_size, len(h_meta["choices"])
                            ),
                            "softmax": nn.Softmax(1),
                            "embedding": nn.Linear(
                                len(h_meta["choices"]), self.input_size
                            )
                        }
                        action_meta = {
                            k: v for k, v in h_meta.items()
                            if k in ["choices"]
                        }
                    elif h_meta["type"] in CONTINUOUS_HYPERPARAM_TYPES:
                        action_head_layers = {
                            "dense": nn.Linear(
                                self.output_size, self.hidden_size
                            ),
                            "mu": nn.Sequential(
                                nn.Linear(self.hidden_size, 1),
                                nn.Softplus()
                            ),
                            "sigma": nn.Sequential(
                                nn.Linear(self.hidden_size, 1),
                                nn.Softplus()
                            ),
                            "embedding": nn.Linear(
                                # one input unit per estimated parameter
                                2, self.input_size,
                            )
                        }
                        action_meta = {
                            k: v for k, v in h_meta.items()
                            if k in ["min", "max"]
                        }
                    exclude_masks = self._exclusion_condition_to_mask(
                        hp_state_space,
                        h_meta,
                        hp_exclude_cond_map.get(hname, None)
                    )
                    hp_key = f"hp_{hname}"
                    self.action_modules[hp_key] = nn.ModuleDict(
                        action_head_layers
                    )
                    self.action_meta[hp_key] = {
                        "action_type": CASHComponent.HYPERPARAMETER,
                        "name": hname,
                        "exclude_masks": exclude_masks,
                        "hyperparameter_type": h_meta["type"],
                        **action_meta,
                    }
                    self.action_index["algo_to_hp"][acomponent].append(hp_key)

    def _exclusion_condition_to_mask(
        self, hp_state_space, hyperparam_meta, exclude
    ):
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

                if hp_state_space[hname]["type"] is HyperparamType.CATEGORICAL:
                    mask = (
                        [1 for _ in hp_state_space[hname]["choices"]]
                        if exclude_values == EXCLUDE_ALL
                        else [
                            int(i in exclude_values) for i in
                            hp_state_space[hname]["choices"]
                        ]
                    )
                    mask = torch.ByteTensor(mask)
                elif (
                    hp_state_space[hname]["type"]
                    in CONTINUOUS_HYPERPARAM_TYPES
                ):
                    if exclude_values == EXCLUDE_ALL:
                        mask = torch.tensor(float("nan"))
                    else:
                        raise ValueError(
                            "can only EXCLUDE_ALL on continuous "
                            "hyperparameters"
                        )
                exclude_masks[choice].update({hname: mask})

        return exclude_masks

    def get_policy_dist(
        self, action, hidden, action_modules, action_meta, mask=None
    ):
        """Get action distribution from policy."""
        rnn_output, hidden = self.micro_action_rnn(action, hidden)
        rnn_output = self.micro_action_dropout(
            self.micro_action_decoder(rnn_output))

        # obtain action probability distribution
        rnn_output = rnn_output.view(rnn_output.shape[0], -1)

        if "softmax" in action_modules:
            logits = action_modules["dense"](rnn_output)
            if mask is not None:
                logger.warning(
                    "applying mask %s to action_classifier %s" %
                    (mask, action_modules)
                )
                logits[mask] = float("-inf")
            action_prob_dist = Categorical(action_modules["softmax"](logits))
        elif all(x in action_modules for x in ["mu", "sigma"]):
            activations = action_modules["dense"](rnn_output).squeeze(0)
            action_prob_dist = Normal(
                action_modules["mu"](activations),
                action_modules["sigma"](activations),
            )
        else:
            raise ValueError(f"action module not recognized: {action_modules}")
        return action_prob_dist, hidden

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

        # maps hyperparameter names to conditional masks
        self._exclude_masks: dict = {}
        # for each algorithm component type, select an algorithm
        if self.mlf_signature is None:
            algo_components = self.a_space.component_dict_from_target_type(
                target_type
            )
        else:
            algo_components = self.a_space.component_dict_from_signature(
                self.mlf_signature
            )

        for atype in algo_components:
            algo_action, action_tensor, hidden = self._decode_action(
                action_tensor, hidden,
                action_index=self.action_index["algo_type"][atype]
            )
            if algo_action is None:
                raise RuntimeError(
                    "selected action for algorithm component cannot be None.\n"
                    "exclude masks: %s\n"
                    "algorithm components: %s\n"
                    "algorithm type: %s" % (
                        self._exclude_masks,
                        algo_components,
                        atype,
                    )
                )
            actions.append(algo_action)

            # each algorithm is associated with a set of hyperparameters
            for hp_action_index in (
                self.action_index["algo_to_hp"][algo_action["choice"]]
            ):
                hp_action, action_tensor, hidden = self._decode_action(
                    action_tensor, hidden, action_index=hp_action_index)
                if hp_action is None:
                    logger.warning(
                        "selected action for hyperparameter index "
                        f"{hp_action_index} for algorithm choice "
                        f"{algo_action['choice']}"
                    )
                    continue
                actions.append(hp_action)
        return value, actions, action_tensor, hidden

    def _get_exclusion_mask(self, action_name):
        if action_name in self._exclude_masks:
            exclude_mask = self._exclude_masks[action_name]
            return exclude_mask.view(1, -1).type(torch.bool)
        else:
            return None

    def _accumulate_exclusion_mask(self, masks):
        for action_name, m in masks.items():
            mask = self._exclude_masks.get(action_name, None)
            if mask is None or torch.isnan(mask).all():
                self._exclude_masks[action_name] = m
            else:
                acc_mask = (m.int() + mask.int()) > 0
                acc_mask = acc_mask.type(torch.ByteTensor)
                self._exclude_masks[action_name] = acc_mask

    def _decode_action(
        self, action_tensor, hidden, action_index
    ):
        action_modules = self.action_modules[action_index]
        action_meta = self.action_meta[action_index]

        mask = self._get_exclusion_mask(action_meta["name"])
        if mask is not None and ((mask == 1).all() or torch.isnan(mask).all()):
            # If all actions are masked, then don't select any choice
            logger.warning(
                f"all actions are masked for action {action_meta}"
            )
            return None, action_tensor, hidden

        action_prob_dist, hidden = self.get_policy_dist(
            action_tensor, hidden, action_modules, action_meta, mask
        )
        action = self._select_action(action_prob_dist, action_meta, mask)
        if action is None:
            return None, action_tensor, hidden

        action_tensor = self.encode_embedding(
            action_index, action["choice_tensor"]
        )
        return action, action_tensor, hidden

    def _select_action(self, action_prob_dist, action_meta, mask):
        """Select action based on action probability distribution."""
        # compute unmasked log probabilitises.
        if isinstance(action_prob_dist, Categorical):
            sampled = action_prob_dist.sample()
            choice_index = sampled.data.item()
            choice = action_meta["choices"][choice_index]
            choice_tensor = action_prob_dist.probs
            choice_meta = {"choices": action_meta["choices"]}
        elif isinstance(action_prob_dist, Normal):
            sampled = action_prob_dist.rsample()
            sampled = torch.clamp(sampled, -1, 1)
            choice_index = None
            choice = sampled.data.item()

            # project sampled choice into min and max range
            mid = (action_meta["max"] - action_meta["min"]) / 2
            choice = mid + (mid * choice)

            # ugly hack, need to figure out more elegant way of handling this
            choice = (
                action_meta["min"]
                if choice < action_meta["min"]
                else action_meta["max"]
                if choice > action_meta["max"]
                else choice
            )

            if action_meta["hyperparameter_type"] is HyperparamType.INTEGER:
                choice = int(choice)
            choice_tensor = torch.cat(
                [action_prob_dist.loc, action_prob_dist.scale]
            ).unsqueeze(0)
            choice_meta = {k: action_meta[k] for k in ["min", "max"]}
        else:
            raise ValueError(
                f"action probability distribution {type(action_prob_dist)} "
                "not recognized."
            )

        # for current choice, accumulate exclusion masks if any
        if action_meta["exclude_masks"] is not None and \
                choice in action_meta["exclude_masks"]:
            for _, masks in action_meta["exclude_masks"].items():
                self._accumulate_exclusion_mask(masks)

        return {
            "action_type": action_meta["action_type"],
            "action_name": action_meta["name"],
            "choice": choice,
            "choice_index": choice_index,
            "choice_tensor": choice_tensor,
            "log_prob": action_prob_dist.log_prob(sampled),
            "entropy": action_prob_dist.entropy(),
            **choice_meta,
        }

    def encode_embedding(self, action_index, choice_tensor):
        """Encode action choice into embedding at input for next action."""
        embedding = self.action_modules[action_index]["embedding"]
        action_embedding = embedding(choice_tensor)
        return action_embedding.view(
            1, action_embedding.shape[0], action_embedding.shape[1]
        )

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
            "action_meta": self.action_meta,
            "action_index": self.action_index,
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
    def load(cls, path, **kwargs):
        """Load saved controller."""
        model_config = torch.load(path, pickle_module=dill)
        controller = cls(**model_config["config"], **kwargs)
        state_dict = controller.state_dict()
        state_dict.update({
            k: v for k, v in model_config["weights"].items()
            if k in state_dict
        })
        controller.load_state_dict(state_dict)
        return controller
