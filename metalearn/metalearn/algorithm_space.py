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

import numpy as np

from collections import OrderedDict

from sklearn.pipeline import Pipeline

from . import components
from .data_types import AlgorithmType, TargetType
from .components.constants import START_TOKEN, END_TOKEN, NONE_TOKEN


SPECIAL_TOKENS = [START_TOKEN, END_TOKEN, NONE_TOKEN]
# ml framework pipeline must have this signature. Can eventually support
# multiple signatures.
CLASSIFIER_MLF_SIGNATURE = [
    AlgorithmType.IMPUTER,
    AlgorithmType.ONE_HOT_ENCODER,
    AlgorithmType.RESCALER,
    AlgorithmType.FEATURE_PREPROCESSOR,
    AlgorithmType.CLASSIFIER
]
REGRESSOR_MLF_SIGNATURE = [
    AlgorithmType.IMPUTER,
    AlgorithmType.ONE_HOT_ENCODER,
    AlgorithmType.RESCALER,
    AlgorithmType.FEATURE_PREPROCESSOR,
    AlgorithmType.REGRESSOR
]
TARGET_TYPE_TO_MLF_SIGNATURE = {
    TargetType.BINARY: CLASSIFIER_MLF_SIGNATURE,
    TargetType.MULTICLASS: CLASSIFIER_MLF_SIGNATURE,
    TargetType.REGRESSION: REGRESSOR_MLF_SIGNATURE,
    TargetType.MULTIREGRESSION: REGRESSOR_MLF_SIGNATURE,
}


class AlgorithmSpace(object):
    """Class that generates machine learning frameworks."""

    ALL_COMPONENTS = [
        AlgorithmType.IMPUTER,
        AlgorithmType.ONE_HOT_ENCODER,
        AlgorithmType.RESCALER,
        AlgorithmType.FEATURE_PREPROCESSOR,
        AlgorithmType.CLASSIFIER,
        AlgorithmType.REGRESSOR
    ]

    def __init__(self,
                 data_preprocessors=None,
                 feature_preprocessors=None,
                 classifiers=None,
                 regressors=None,
                 with_start_token=False,
                 with_end_token=False,
                 with_none_token=False,
                 hyperparam_with_start_token=False,
                 hyperparam_with_end_token=False,
                 hyperparam_with_none_token=False,
                 random_state=None):
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
        self.regressors = get_regressors() if regressors is None \
            else regressors
        # TODO: assess whether these tokens are necessary
        self.with_start_token = with_start_token
        self.with_end_token = with_end_token
        self.with_none_token = with_none_token
        self.hyperparam_with_start_token = hyperparam_with_start_token
        self.hyperparam_with_end_token = hyperparam_with_end_token
        self.hyperparam_with_none_token = hyperparam_with_none_token
        self.random_state = random_state
        np.random.seed(self.random_state)

    @property
    def components(self):
        """Concatenate all components into a single list."""
        components = self.data_preprocessors + \
            self.feature_preprocessors + \
            self.classifiers + \
            self.regressors
        if self.with_start_token:
            components += [START_TOKEN]
        if self.with_end_token:
            components += [END_TOKEN]
        if self.with_none_token:
            components += [NONE_TOKEN]
        return components

    @property
    def n_components(self):
        """Return number of components in the algorithm space."""
        return len(self.components)

    def component_dict_from_signature(self, signature):
        """Return dictionary of algorithm types and list of components.

        :param list[str] signature: ML framework signature indicating the
            ordering of algorithm components to form a sklearn Pipeline.
        """
        return OrderedDict([
            (component_type, self.get_components(component_type))
            for component_type in signature])

    def component_dict_from_target_type(self, target_type):
        """Get algorithm components based on target type.

        :param data_types.TargetType target_type: get the MLF signature for
            this target.
        """
        return self.component_dict_from_signature(
            TARGET_TYPE_TO_MLF_SIGNATURE[target_type])

    def get_components(self, component_type):
        """Get all components of a particular type.

        :param str component_type: type of algorithm
        :returns: list of components of `component_type`
        :rtype: list[AlgorithmComponent]
        """
        return [c for c in self.components if c not in SPECIAL_TOKENS and
                c.component_type == component_type]

    def h_state_space(self, components, with_none_token=False):
        """Get hyperparameter state space by components.

        :param list[AlgorithmComponent] components: list of components
        :returns: OrderedDict of hyperparameters
        :rtype: dict[str -> list]
        """
        hyperparam_states = OrderedDict()
        for c in components:
            if c not in SPECIAL_TOKENS and c.hyperparameters is not None:
                hyperparam_states.update(
                    c.hyperparameter_state_space(with_none_token))
        return hyperparam_states

    def h_exclusion_conditions(self, components):
        """Get the conditional map of which hyperparameters go together."""
        exclude_conditions = OrderedDict()
        for c in components:
            if c and c.hyperparameters is not None:
                exclude_conditions.update(
                    c.hyperparameter_exclusion_conditions())
        return exclude_conditions

    def sample_component(self, component_type):
        """Sample a component of a particular type.

        :param str component_type: type of algorithm, one of
            {"one_hot_encoder", "imputer", "rescaler", "feature_preprocessor",
             "classifier", "regressor"}
        :returns: a sampled algorithm component of type `component_type`.
        :rtype: AlgorithmComponent
        """
        component_subset = self.get_components(component_type)
        return component_subset[np.random.randint(len(component_subset))]

    def sample_components_from_signature(self, signature):
        """Sample algorithm components from ML signature.

        :param list[str] signature: ML framework signature indicating the
            ordering of algorithm components to form a sklearn Pipeline.
        """
        return [self.sample_component(component_type)
                for component_type in signature]

    def sample_ml_framework(self, signature, task_metadata=None):
        """Sample a random ML framework from the algorithm space.

        :param list[str] signature: ML framework signature indicating the
            ordering of algorithm components to form a sklearn Pipeline.
        :param int|None random_state: provide random state, which determines
            the ML framework sampled.
        :param dict[str, any] task_metadata: constraints imposed by the
            environment on the hyperparameter space.
        """
        components = self.sample_components_from_signature(signature)
        framework_hyperparameters = {}
        for a in components:
            framework_hyperparameters.update(
                a.sample_hyperparameter_state_space())
        return self.create_ml_framework(
            components, hyperparameters=framework_hyperparameters,
            task_metadata=task_metadata)

    def create_ml_framework(
            self, components, memory=None, hyperparameters=None,
            task_metadata=None):
        """Create ML framework, in this context an sklearn pipeline object.

        :param list[AlgorithmComponent] components: A list of algorithm
            components with which to create an ML framework.
        :param str|None memory: path to caching directory in which to store
            fitten transformers of the sklearn.Pipeline. If None, no caching
            is done
        :param dict[str, str] hyperparameters: picked by the automl system.
        :param dict[str, any] task_metadata: constraints imposed by the
            environment on the hyperparameter space.
        """
        steps = []
        hyperparameters = {} if hyperparameters is None else hyperparameters
        task_metadata = {} if task_metadata is None else task_metadata
        for component in components:
            steps.append(
                (component.name, component(
                    categorical_features=task_metadata.get(
                        "categorical_features", None),
                    continuous_features=task_metadata.get(
                        "continuous_features", None)
                    ))
            )
            hyperparameters.update(component.get_constant_hyperparameters())
        ml_framework = Pipeline(memory=memory, steps=steps)
        ml_framework.set_params(**hyperparameters)
        return ml_framework

    def _combine_dicts(self, dicts):
        combined_dicts = {}
        for d in dicts:
            for h in d:
                combined_dicts.update(h)
        return combined_dicts

    def set_random_state(self, random_state):
        np.random.seed(random_state)

    @property
    def config(self):
        return {
            "data_preprocessors": self.data_preprocessors,
            "feature_preprocessors": self.feature_preprocessors,
            "classifiers": self.classifiers,
            "regressors": self.regressors,
            "with_start_token": self.with_start_token,
            "with_end_token": self.with_end_token,
            "with_none_token": self.with_none_token,
            "hyperparam_with_start_token": self.hyperparam_with_start_token,
            "hyperparam_with_end_token": self.hyperparam_with_end_token,
            "hyperparam_with_none_token": self.hyperparam_with_none_token,
            "random_state": self.random_state,
        }

    def __eq__(self, other):
        return self.config == other.config


def get_data_preprocessors():
    """Get all data preprocessors in structured algorithm space."""
    return [
        components.data_preprocessors.simple_imputer(),
        components.data_preprocessors.one_hot_encoder(),
        components.data_preprocessors.minmax_scaler(),
        components.data_preprocessors.standard_scaler(),
        components.data_preprocessors.robust_scaler(),
        components.data_preprocessors.normalizer(),
    ]


def get_feature_preprocessors():
    """Get all feature preprocessors in structured algorithm space."""
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
        components.feature_preprocessors.variance_threshold_filter(),
    ]


def get_classifiers():
    """Get all classifiers in structured algorithm space."""
    return [
        components.classifiers.adaboost(),
        components.classifiers.decision_tree(),
        components.classifiers.gaussian_naive_bayes(),
        components.classifiers.gradient_boosting(),
        components.classifiers.k_nearest_neighbors(),
        components.classifiers.logistic_regression(),
        components.classifiers.multinomial_naive_bayes(),
        components.classifiers.random_forest_classifier(),
        components.classifiers.rbf_gaussian_process_classifier(),
        components.classifiers.support_vector_classifier_linear(),
        components.classifiers.support_vector_classifier_nonlinear(),
    ]


def get_regressors():
    """Get all classifiers in structured algorithm space."""
    return [
        components.regressors.adaboost_regression(),
        components.regressors.ard_regression(),
        components.regressors.bayesian_ridge_regression(),
        components.regressors.decision_tree_regression(),
        components.regressors.k_nearest_neighbors_regression(),
        components.regressors.lasso_regression(),
        components.regressors.random_forest_regression(),
        components.regressors.rbf_gaussian_process_regression(),
        components.regressors.ridge_regression(),
        components.regressors.support_vector_regression_linear(),
        components.regressors.support_vector_regression_nonlinear(),
    ]
