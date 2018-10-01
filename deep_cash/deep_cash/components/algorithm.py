"""Define algorithm component."""

from collections import OrderedDict

import numpy as np

import itertools

from . import constants


class AlgorithmComponent(object):
    """A component of a machine learning framework F."""

    def __init__(
            self,
            name,
            component_class,
            component_type=None,
            hyperparameters=None,
            constant_hyperparameters=None,
            env_dep_hyperparameters=None):
        """Initialize an AlgorithmComponent.

        :param str name: name of component.
        :param object component_class: of type sklearn.BaseEstimator
        :param str component_type: type of algorithm.
        :param list[Hyperparameters]|None hyperparameters: list of
            Hyperparameter objects, which specify algorithms' hyperparameter
            space.
        :param dict constant_hyperparameters: a set of hyperparameters that
            shouldn't be picked by
        :param dict env_dep_hyperparameters: a set of hyperparameters in the
            algorithm component that are dependent on the data environment.
            For now these hyperparameters are set by the data environment and
            are not tuned by the controller. This may change in the future.
        """
        if component_type not in constants.ALGORITHM_TYPES:
            raise ValueError("%s is not a valid algorithm type: choose %s" % (
                component_type, constants.ALGORITHM_TYPES))
        self.name = name
        self.component_class = component_class
        self.component_type = component_type
        self.hyperparameters = hyperparameters
        self.constant_hyperparameters = {} if \
            constant_hyperparameters is None else constant_hyperparameters
        self.env_dep_hyperparameters = {} if env_dep_hyperparameters is None \
            else env_dep_hyperparameters

    def __call__(self):
        """Instantiate the algorithm.

        When instantiating the algorithm, optionally supply a data-envirionment
        specific set of hyperparameters.
        """
        return self.component_class(**self.constant_hyperparameters)

    def env_dep_hyperparameter_name_space(self):
        """Return a dictionary of hyperparameters in algorithm name space."""
        return {
            "%s__%s" % (self.name, h): value
            for h, value in self.env_dep_hyperparameters.items()
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
            ("%s__%s" % (self.name, h.hname),
                h.get_state_space(with_none_token))
            for h in self.hyperparameters])

    def hyperparameter_iterator(self):
        """Return a generator of all possible hyperparameter combinations."""
        expanded_state_space = []
        for key, values in self.hyperparameter_state_space().items():
            expanded_state_space.append([(key, v) for v in values])
        return (
            dict(hsetting) for hsetting in
            list(itertools.product(*expanded_state_space)))

    def sample_hyperparameter_state_space(self):
        """Return a random sample from the hyperparameter state space."""
        settings = {}
        for key, values in self.hyperparameter_state_space().items():
            settings[key] = values[np.random.randint(len(values))]
        return settings

    def __repr__(self):
        return "<AlgorithmComponent: \"%s\">" % self.name
