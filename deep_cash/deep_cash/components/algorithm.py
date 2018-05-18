"""Define algorithm component."""

from collections import OrderedDict

import numpy as np

import itertools

from . import constants


class AlgorithmComponent(object):
    """A component of a machine learning framework F."""

    def __init__(self, aname, aclass, atype=None, hyperparameters=None,
                 env_dep_hyperparameters=None):
        """Initialize an AlgorithmComponent.

        :param str aname: name of component.
        :param object aclass: of type sklearn.BaseEstimator
        :param str atype: type of algorithm.
        :param list[Hyperparameters]|None hyperparameters: list of
            Hyperparameter objects, which specify algorithms' hyperparameter
            space.
        :param dict env_dep_hyperparameters: a set of hyperparameters in the
            algorithm component that are dependent on the data environment.
            For now these hyperparameters are set by the data environment and
            are not tuned by the controller. This may change in the future.
        """
        if atype not in constants.ALGORITHM_TYPES:
            raise ValueError("%s is not a valid algorithm type: choose %s" % (
                atype, constants.ALGORITHM_TYPES))
        self.aname = aname
        self.aclass = aclass
        self.atype = atype
        self.hyperparameters = hyperparameters

        self.env_dep_hyperparameters = {} if env_dep_hyperparameters is None \
            else env_dep_hyperparameters

    def __call__(self, env_dep_hyperparameters=None):
        """Instantiate the algorithm.

        When instantiating the algorithm, optionally supply a data-envirionment
        specific set of hyperparameters.
        """
        return self.aclass(
            **self.env_dep_hyperparameters if env_dep_hyperparameters is None
            else env_dep_hyperparameters)

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
