"""Module for creating a structured algorithm environment.

The structured algorithm space is different from the `algorithm_env.py`
module because here we specify an interface to generate a machine learning
framework F, which is a sequence of algorithms (estimators/transformers) and
their associated hyperparameters. This interface is inspired by this paper
on automated machine learning:

papers.nips.cc/paper/5872-efficient-and-robust-automated-machine-learning.pdf

The sequence follows the general structure

.-------------------.    .----------------------.    .----------------------.
| data preprocessor | -> | feature preprocessor | -> | classifier/regressor |
.-------------------.    .----------------------.    .----------------------.

where there can be n data preprocessors (imputation, scaling, encoding,
filtering).
"""

import itertools
import numpy as np

from collections import OrderedDict

from sklearn.pipeline import Pipeline

from . import components
from .components import constants
from .components.constants import START_TOKEN, END_TOKEN, NONE_TOKEN
from . import utils


SPECIAL_TOKENS = [START_TOKEN, END_TOKEN, NONE_TOKEN]
# ml framework pipeline must have this signature. Can eventually support
# multiple signatures.
ML_FRAMEWORK_SIGNATURE = [
    constants.ONE_HOT_ENCODER,
    constants.IMPUTER,
    constants.RESCALER,
    constants.FEATURE_PREPROCESSOR,
    constants.CLASSIFIER
]


class AlgorithmSpace(object):
    """A class that generates machine learning frameworks."""

    # bonus rewards for proposing valid ml frameworks and hyperparameter
    # settings.
    VALID_MLF_BONUS = 25
    VALID_MLFH_BONUS = 50

    # reward for getting a single hyperparameter correct, in the training
    # regime that every hyperparameter selection action yields a reward.
    CORRECT_HYPERPARAMETER_REWARD = 1
    INCORRECT_HYPERPARAMETER_REWARD = -1
    ALL_CORRECT_MULTIPLIER = 5
    ALL_INCORRECT_MULTIPLIER = 5

    N_COMPONENT_TYPES = len(ML_FRAMEWORK_SIGNATURE)
    ML_FRAMEWORK_SIGNATURE = ML_FRAMEWORK_SIGNATURE

    def __init__(self, data_preprocessors=None, feature_preprocessors=None,
                 classifiers=None, with_start_token=True,
                 with_end_token=False, with_none_token=False,
                 hyperparam_with_start_token=True,
                 hyperparam_with_end_token=False,
                 hyperparam_with_none_token=True):
        """Initialize a structured algorithm environment.

        :param list[AlgorithmComponent]|None data_preprocessors: algorithm
            components for data preprocessing
        :param list[AlgorithmComponent]|None feature_preprocessors: algorithm
            components for feature preprocessing
        :param list[AlgorithmComponent]|None classifiers: algorithm
            components for classification
        """
        self.data_preprocessors = get_data_preprocessors() \
            if data_preprocessors is None else data_preprocessors
        self.feature_preprocessors = get_feature_preprocessors() \
            if feature_preprocessors is None else feature_preprocessors
        self.classifiers = get_classifiers() if classifiers is None \
            else classifiers
        self.with_start_token = with_start_token
        self.with_end_token = with_end_token
        self.with_none_token = with_none_token
        self.hyperparam_with_start_token = hyperparam_with_start_token
        self.hyperparam_with_end_token = hyperparam_with_end_token
        self.hyperparam_with_none_token = hyperparam_with_none_token
        self._logger = utils.init_logging(__file__)

    @property
    def components(self):
        """Concatenate all components into a single list."""
        components = self.data_preprocessors + \
            self.feature_preprocessors + \
            self.classifiers
        if self.with_start_token:
            components += [START_TOKEN]
        if self.with_end_token:
            components += [END_TOKEN]
        if self.with_none_token:
            components += [NONE_TOKEN]
        return components

    @property
    def hyperparameter_name_space(self):
        """Return all hyperparameters for all components in the space."""
        return self.h_name_space(self.components)

    @property
    def hyperparameter_state_space(self):
        """Return all hyperparameter name-value pairs."""
        return self.h_state_space(self.components)

    @property
    def hyperparameter_state_space_flat(self):
        """Return all hyperparameter values in flat structure.

        In the following form:
        {
            "Algorithm__hyperparameter__value_0": value_0,
            "Algorithm__hyperparameter__value_1": value_1,
            ...
            "Algorithm__hyperparameter__value_n": value_n,
        }
        """
        hyperparam_values = OrderedDict()
        for name, space in self.hyperparameter_state_space.items():
            for i, value in enumerate(space):
                hyperparam_values["%s__state_%d" % (name, i)] = value
        if self.hyperparam_with_start_token:
            hyperparam_values["START_TOKEN"] = START_TOKEN
        if self.hyperparam_with_end_token:
            hyperparam_values["END_TOKEN"] = END_TOKEN
        if self.hyperparam_with_none_token:
            hyperparam_values["NONE_TOKEN"] = NONE_TOKEN
        return hyperparam_values

    @property
    def hyperparameter_state_space_values(self):
        return list(self.hyperparameter_state_space_flat.values())

    @property
    def hyperparameter_state_space_keys(self):
        return list(self.hyperparameter_state_space_flat.keys())

    @property
    def start_token_index(self):
        """Return index of the start of sequence token."""
        return self.components.index(START_TOKEN) if self.with_start_token \
            else None

    @property
    def end_token_index(self):
        """Return index of the end of sequence token."""
        return self.components.index(END_TOKEN) if self.with_end_token \
            else None

    @property
    def none_token_index(self):
        """Return index of the none token."""
        return self.components.index(NONE_TOKEN) if self.with_none_token \
            else None

    @property
    def h_start_token_index(self):
        """Return index of hyperparameter start token."""
        return self.hyperparameter_state_space_values.index(START_TOKEN) if \
            self.with_start_token else None

    @property
    def n_components(self):
        """Return number of components in the algorithm space."""
        return len(self.components)

    @property
    def n_hyperparameter_names(self):
        """Return number of hyperparameter"""
        return len(self.hyperparameter_name_space)

    @property
    def n_hyperparameters(self):
        """Return number of hyperparameter"""
        return len(self.hyperparameter_state_space_flat)

    def sample_components_from_signature(self, signature=None):
        signature = self.ML_FRAMEWORK_SIGNATURE if signature is None \
            else signature
        return [self.sample_component(atype) for atype in signature]

    def component_dict_from_signature(self, signature=None):
        signature = self.ML_FRAMEWORK_SIGNATURE if signature is None \
            else signature
        return OrderedDict([
            (atype, self.get_components(atype)) for atype in signature])

    def get_components(self, atype):
        """Get all components of a particular type.

        :param str atype: type of algorithm
        :returns: list of components of `atype`
        :rtype: list[AlgorithmComponent]
        """
        return [c for c in self.components if c not in SPECIAL_TOKENS and
                c.atype == atype]

    def h_name_space(self, components):
        """Get hyperparameter name space by components.

        :param list[AlgorithmComponent] components: list of components
        :returns: list of hyperparameter names
        :rtype: list[str]
        """
        hyperparam_names = []
        for c in components:
            if c not in SPECIAL_TOKENS and c.hyperparameters is not None:
                hyperparam_names.extend(c.hyperparameter_name_space())
        return hyperparam_names

    def h_state_space(self, components, with_none_token=False):
        """Get hyperparameter state space by components.

        :param list[AlgorithmComponent] components: list of components
        :returns: list of hyperparameter names
        :rtype: list[str]
        """
        hyperparam_states = OrderedDict()
        for c in components:
            if c not in SPECIAL_TOKENS and c.hyperparameters is not None:
                hyperparam_states.update(
                    c.hyperparameter_state_space(with_none_token))
        return hyperparam_states

    def h_value_index(self, hyperparameter_name):
        """Check whether a hyperparameter value index is correct."""
        return [
            i for i, (k, v) in enumerate(
                self.hyperparameter_state_space_flat.items())
            if k.startswith(hyperparameter_name)]

    def sample_component(self, atype):
        """Sample a component of a particular type.

        :param str atype: type of algorithm, one of {"one_hot_encoder",
            "imputer", "rescaler", "feature_preprocessor", "classifier",
            "regressor"}
        :returns: a sampled algorithm component of type `atype`.
        :rtype: AlgorithmComponent
        """
        component_subset = self.get_components(atype)
        return component_subset[np.random.randint(len(component_subset))]

    def sample_ml_framework(self, random_state=None):
        """Sample a random ML framework from the algorithm space.

        :param int|None random_state: provide random state, which determines
            the ML framework sampled.
        """
        components = self.sample_components_from_signature()
        framework_hyperparameters = {}
        for a in components:
            framework_hyperparameters.update(
                a.sample_hyperparameter_state_space())
        return self.create_ml_framework(
            components, hyperparameters=framework_hyperparameters)

    def framework_iterator(self):
        """Return a generator of all algorithm and hyperparameter combos.

        This is potentially a huge space, creating a generator that yields
        a machine learning framework (sklearn.Pipeline object) based on all
        possible estimator combinations and all possible hyperparameter
        combinations of those estimators.
        """
        return (
            self.create_ml_framework(
                component_list,
                hyperparameters=self._combine_dicts(hyperparam_list_dicts))
            for component_list in itertools.product(
                self.sample_components_from_signature())
            for hyperparam_list_dicts in itertools.product(
                [c.hyperparameter_iterator() for c in component_list])
        )

    def create_ml_framework(
            self, components, memory=None, hyperparameters=None):
        """Create ML framework, in this context an sklearn pipeline object.

        :param list[AlgorithmComponent] components: A list of algorithm
            components with which to create an ML framework.
        :param str|None memory: path to caching directory in which to store
            fitten transformers of the sklearn.Pipeline. If None, no caching
            is done
        """
        # TODO: call a() instead of a.aclass()
        steps = []
        env_dep_hyperparameters = {}
        for a in components:
            steps.append((a.aname, a()))
            env_dep_hyperparameters.update(
                a.env_dep_hyperparameter_name_space())
        ml_framework = Pipeline(memory=memory, steps=steps)
        if hyperparameters is not None:
            h = hyperparameters.copy()
            h.update(env_dep_hyperparameters)
            ml_framework.set_params(**h)
        return ml_framework

    def set_ml_framework_params(self, ml_framework, hyperparameters):
        """Set parameters of ML framework.

        WARNING: this will over-ride the env_dep_hyperparameters if the
        components in the ml_framework.

        :param sklearn.Pipeline ml_framework: a ml framework.
        :param dict framework_hyperparameters: hyperparameters of the pipeline.
        """
        hyperparameters = OrderedDict([
            (k, v) for k, v in hyperparameters.items()
            if v != NONE_TOKEN])
        return ml_framework.set_params(**hyperparameters)

    def check_ml_framework(self, pipeline, sig_check=1):
        """Check if the steps in ML framework form a valid pipeline.

        WARNING: this will over-ride the env_dep_hyperparameters if the
        components in the ml_framework.
        """
        # TODO: add more structure to an ml framework:
        # Data Preprocessor > Feature Preprocessor > Classifier
        pipeline = [p for p in pipeline if p not in SPECIAL_TOKENS]
        try:
            assert hasattr(pipeline[-1].aclass, "predict")
            assert len(pipeline) == len(ML_FRAMEWORK_SIGNATURE)
            assert [a.atype for a in pipeline][:sig_check] == \
                ML_FRAMEWORK_SIGNATURE[:sig_check]
            return self.create_ml_framework(pipeline, memory=None)
        except Exception:
            return None

    def evaluate_hyperparameters(
            self, ml_framework, hyperparameters, h_value_indices):
        none_index = self.hyperparameter_state_space_keys.index("NONE_TOKEN")
        errors = {}
        rewards = []
        n_correct = 0
        for h, idx in h_value_indices.items():
            if idx not in self.h_value_index(h) and idx != none_index:
                rewards.append(self.INCORRECT_HYPERPARAMETER_REWARD)
                # for error analysis
                errors.update({h: hyperparameters[h]})
            else:
                n_correct += 1
                rewards.append(self.CORRECT_HYPERPARAMETER_REWARD)
        if n_correct == len(hyperparameters):
            rewards = [r * self.ALL_CORRECT_MULTIPLIER for r in rewards]
        elif n_correct == 0:
            rewards = [r * self.ALL_INCORRECT_MULTIPLIER for r in rewards]
        return rewards, n_correct, errors

    def check_hyperparameters(
            self, ml_framework, hyperparameters, h_value_indices):
        """Check if the selected hyperparameters are valid."""
        none_index = self.hyperparameter_state_space_keys.index("NONE_TOKEN")
        try:
            for h, idx in h_value_indices.items():
                if idx not in self.h_value_index(h) and idx != none_index:
                    return None
            return self.set_ml_framework_params(
                ml_framework, hyperparameters)
        except Exception:
            self._logger.exception("HYPERPARAMETER CHECK FAIL")
            return None

    def _combine_dicts(self, dicts):
        combined_dicts = {}
        for d in dicts:
            combined_dicts.update(d)
        return combined_dicts


def get_data_preprocessors():
    """Get all data preprocessors in structured algorithm environment."""
    return [
        components.data_preprocessors.imputer(),
        components.data_preprocessors.one_hot_encoder(),
        components.data_preprocessors.variance_threshold_filter(),
        components.data_preprocessors.minmax_scaler(),
        components.data_preprocessors.standard_scaler(),
        components.data_preprocessors.robust_scaler(),
        components.data_preprocessors.normalizer(),
    ]


def get_feature_preprocessors():
    """Get all feature preprocessors in structured algorithm environment."""
    return [
        components.feature_preprocessors.fast_ica(),
        components.feature_preprocessors.feature_agglomeration(),
        components.feature_preprocessors.kernel_pca(),
        components.feature_preprocessors.rbf_sampler(),
        components.feature_preprocessors.nystroem_sampler(),
        components.feature_preprocessors.pca(),
        components.feature_preprocessors.polynomial_features(),
        components.feature_preprocessors.random_trees_embedding(),
        components.feature_preprocessors.truncated_svd(),
    ]


def get_classifiers():
    """Get all classifiers in structured algorithm environment."""
    return [
        components.classifiers.logistic_regression(),
        components.classifiers.gaussian_naive_bayes(),
        components.classifiers.decision_tree(),
    ]
