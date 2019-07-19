"""Define algorithm component."""

import itertools

import numpy as np

from collections import OrderedDict
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from typing import List, Tuple, Any, Union

from ..data_types import AlgorithmType


Hyperparameters = List[List[Tuple[str, Any]]]

# this variable indicates that a hyperparameter should be completely ignored.
EXCLUDE_ALL = "ALL"


def _check_hyperparameters(
        hyperparameters: Hyperparameters,
        exclusion_conditions: dict) -> Union[dict, None]:
    exclude_cases = {}
    for key, value in hyperparameters:

        if key in exclude_cases and value in exclude_cases[key]:
            # early return: the hyperparameter setting is invalid if any of
            # the selected values are specified in the exclude cases
            return None

        exclude_dict = exclusion_conditions.get(key, None)
        if exclude_dict is None:
            continue
        elif value not in exclude_dict:
            continue
        exclude_cases.update(exclude_dict[value])

    return dict(hyperparameters)


class AlgorithmComponent(object):
    """A component of a machine learning framework."""

    def __init__(
            self,
            name,
            component_class,
            component_type=None,
            initialize_component=None,
            hyperparameters=None,
            constant_hyperparameters=None,
            exclusion_conditions=None):
        """Initialize an AlgorithmComponent.

        :param str name: name of component.
        :param object component_class: of type sklearn.BaseEstimator
        :param str component_type: type of algorithm.
        :param callable|None initialize_component: a function invoked in the
            __call__ method that's responsible for instantiating the
            component_class, with the following signature:

            (component_class: EstimatorClass,
             categorical_features: List[int],
             continuous_features: List[int]) -> EstimatorObject

            This function is called in the __call__ method. If None, then
            component_class Estimator is initialized with all of its defaults.

        :param list[Hyperparameters]|None hyperparameters: list of
            Hyperparameter objects, which specify algorithms' hyperparameter
            space.

            The MetaLearnController interprets `hyperparameters` as a DAG where
            at each node represents a hyperparameter in algorithm space.
            At each node, the agent selects a hyperparameter value (currently
            these can only be a discrete set of values). The agent then
            continues to traverse the graph until all hyperparameters have
            been selected.
        :param dict constant_hyperparameters: a set of hyperparameters that
            shouldn't be picked by the controller
        :param dict[str -> dict[str -> list]] exclusion_conditions: a
            dictionary specifying the which of the subsequent hyperparameters
            should be excluded conditioned on picking a particular
            hyperparameter value. For example:

            {
                "hyperparam1": {
                    "h1_value1": {
                        "hyperparam2": ["h2_value1", "h2_value2"]
                    }
                },
                "hyperparam2": {
                    "h2_value_3": {
                        "hyperparam3": ["h3_value1"]
                    }
                }
            }

            The exclusion criteria are executed in the order specified by
            `hyperparameters`. In the example, if the agent selects
            "h1_value1", then, when it comes to the agent selecting a value
            for `hyperparam2`, the agent will only be able to sample from
            the action probabilities that are not `"h2_value1"` or
            `"h2_value2"`.

            These "action probability masks" prevent the agent from selecting
            actions that the environment (in this case the AlgorithmSpace)
            knows is an invalid combination of hyperparameters.

        """
        if component_type not in AlgorithmType:
            raise ValueError("%s is not a valid algorithm type: choose %s" % (
                component_type, list(AlgorithmType)))
        self.name = name
        self.component_class = component_class
        self.initialize_component = initialize_component
        self.component_type = component_type
        self.hyperparameters = hyperparameters
        self.constant_hyperparameters = {} if \
            constant_hyperparameters is None else constant_hyperparameters
        # TODO: consider making a class for exclusion conditions
        self.exclusion_conditions = exclusion_conditions
        # TODO: implement checker for initialize_component function. Make
        #       sure that the estimator portion of the `transformer` arg
        #       is an Estimator

    def __call__(self, categorical_features=None, continuous_features=None):
        """Instantiate the algorithm object.

        :param dict kwargs: keyword arguments to pass into initialize_component
        """
        if self.initialize_component is None:
            return self.component_class()

        estimator = self.initialize_component(
            self.component_class,
            [] if categorical_features is None else categorical_features,
            [] if continuous_features is None else continuous_features)
        if not isinstance(estimator, BaseEstimator):
            raise TypeError(
                "expected output of `initialize_component` to be a "
                f"BaseEstimator, found {type(estimator)}")
        return estimator

    def get_constant_hyperparameters(self):
        """Get environment-dependent hyperparameters."""
        return {
            "%s__%s" % (self.name, h): value
            for h, value in self.constant_hyperparameters.items()
        }

    def hyperparameter_name_space(self):
        """Return list of hyperparameter names."""
        if self.hyperparameters is None:
            return None
        return ["%s__%s" % (self.name, h.hname) for h in self.hyperparameters]

    def hyperparameter_state_space(self, with_none_token=False):
        """Return dict of hyperparameter space."""
        if self.hyperparameters is None:
            return OrderedDict()
        return OrderedDict([
            (n, h.get_state_space(with_none_token)) for n, h in
            zip(self.hyperparameter_name_space(), self.hyperparameters)])

    def hyperparameter_iterator(self):
        """Return a generator of all possible hyperparameter combinations."""

        expanded_state_space = []
        for key, values in self.hyperparameter_state_space().items():
            expanded_state_space.append([(key, v) for v in values])
        return filter(None, (
            _check_hyperparameters(
                hsetting, self.hyperparameter_exclusion_conditions())
            for hsetting in itertools.product(*expanded_state_space)))

    def hyperparameter_exclusion_conditions(self):
        """Get the conditional map of which hyperparameters go together."""
        if self.hyperparameters is None or self.exclusion_conditions is None:
            return OrderedDict()

        # TODO: make sure that keys are actually hyperparameter names
        def format_exclusion_conditions(conds):
            return {h: {"%s__%s" % (self.name, k): v for k, v in ex.items()}
                    for h, ex in conds.items()}

        return OrderedDict([
            ("%s__%s" % (self.name, hname),
                format_exclusion_conditions(exclusion_conditions))
            for hname, exclusion_conditions
            in self.exclusion_conditions.items()])

    def sample_hyperparameter_state_space(self):
        """Return a random sample from the hyperparameter state space."""
        # TODO: incorporate the exclusion criteria in sampling the state space
        # such that only valid combinations of hyperparameters are given.
        settings = {}
        for key, values in self.hyperparameter_state_space().items():
            settings[key] = values[np.random.randint(len(values))]
        return settings

    def __repr__(self):
        return "<AlgorithmComponent: \"%s\">" % self.name

    def __hash__(self):
        return hash((self.name, self.component_class, self.component_type))

    def __eq__(self, other):
        return hash(self) == hash(other)
