"""Define algorithm component."""

from collections import OrderedDict

import numpy as np

import itertools


class AlgorithmComponent(object):
    """A component of a machine learning framework F."""

    def __init__(self, aname, aclass, hyperparameters=None):
        """Initialize an AlgorithmComponent.

        :param str aname: name of component.
        :param object aclass: of type sklearn.BaseEstimator
        :param list[Hyperparameters]|None hyperparameters: list of
            Hyperparameter objects, which specify algorithms' hyperparameter
            space.
        """
        self.aname = aname
        self.aclass = aclass
        self.hyperparameters = hyperparameters

    def hyperparameter_name_space(self):
        """Return list of hyperparameter names.

        TODO: make this a property with the @property decorator
        """
        return ["%s__%s" % (self.aname, h.hname) for h in self.hyperparameters]

    def hyperparameter_state_space(self):
        """Return dict of hyperparameter space.

        TODO: make this a property with the @property decorator
        """
        if self.hyperparameters is None:
            return OrderedDict()
        return OrderedDict([
            ("%s__%s" % (self.aname, h.hname), h.get_state_space())
            for h in self.hyperparameters])

    def hyperparameter_iterator(self):
        """Return a generator of all possible hyperparameter combinations."""
        expanded_state_space = []
        for key, values in self.hyperparameter_state_space().items():
            expanded_state_space.append([(key, v) for v in values])
        return (
            dict(hsetting) for hsetting in
            list(itertools.product(*expanded_state_space)))

    def sample_hyperparameter_state_space(self, random_state=None):
        """Return a random sample from the hyperparameter state space."""
        settings = {}
        np.random.seed(random_state)
        for key, values in self.hyperparameter_state_space().items():
            settings[key] = values[np.random.randint(len(values))]
        return settings

    def __repr__(self):
        return "<AlgorithmComponent: \"%s\">" % self.aname
