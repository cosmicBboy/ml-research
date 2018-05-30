"""Module for abstracting hyperparameters.

Meant to be a lighter-weight version of https://github.com/automl/ConfigSpace,
which incorporates conditional logic and more features, this API only
enumerates the state space of Hyperparameters without enforcing any rules
about how hyperparameters interact with each other. This will presumably be
the job of the Controller RNN.
"""

import itertools
import numpy as np

from . import constants

N_VALUES = 5  # number of values to generate for int/float state spaces


class HyperparameterBase(object):

    def __init__(self, hname, state_space, default):
        """Create hyperparameter base class."""
        self.hname = hname
        self._state_space = list(state_space)
        self.default = default

    def __repr__(self):
        return "<%s: \"%s\">" % (type(self).__name__, self.hname)

    def get_state_space(self, with_none_token=False):
        if self.default in self._state_space:
            state_space = self._state_space
        else:
            state_space = self._state_space + [self.default]
        if with_none_token:
            state_space.append(constants.NONE_TOKEN)
        return state_space


class CategoricalHyperparameter(HyperparameterBase):

    def __init__(self, hname, state_space, default):
        super().__init__(hname, state_space, default)


class NumericalHyperparameter(HyperparameterBase):

    def __init__(self, hname, min, max, dtype, default, log, n):
        self.min = dtype(min)
        self.max = dtype(max)
        self.dtype = dtype
        self.n = n
        self.log = log
        super().__init__(hname, self._init_state_space(), default)

    def _init_state_space(self):
        if self.log:
            # evenly distributed in intervals in log space
            space_func = np.geomspace
        else:
            space_func = np.linspace
        return np.sort(
            space_func(self.min, self.max, self.n, dtype=self.dtype))


class UniformIntHyperparameter(NumericalHyperparameter):

    def __init__(self, hname, min, max, default, log=False, n=N_VALUES):
        super().__init__(hname, min, max, int, default, log, n=n)


class UniformFloatHyperparameter(NumericalHyperparameter):

    def __init__(self, hname, min, max, default, log=False, n=N_VALUES):
        super().__init__(hname, min, max, float, default, log, n=n)


class TuplePairHyperparameter(HyperparameterBase):
    """Tuple Pair Hyperparameter class.

    For hyperparameters in the form: `(x, y)`
    """

    def __init__(self, hname, hyperparameters, default):
        self.hyperparameters = hyperparameters
        super().__init__(hname, self._init_state_space(), default)

    def _init_state_space(self):
        return list(
            itertools.product(
                *[h.get_state_space() for h in self.hyperparameters]))


class TupleRepeatingHyperparameter(HyperparameterBase):
    """Tuple Pair Hyperparameter class.

    For hyperparameters in the form: `(x0, x1, ... , xn)`
    """

    def __init__(self, hname, hyperparameter, max_nrepeats, default):
        self.hyperparameter = hyperparameter
        self.max_nrepeats = max_nrepeats
        super().__init__(hname, self._init_state_space(), default)

    def _state_space_n(self, nrepeats):
        return list(
            itertools.product(
                *[self.hyperparameter.get_state_space()
                  for _ in range(nrepeats)]))

    def _init_state_space(self):
        if self.max_nrepeats == 1:
            return [(i,) for i in self.hyperparameter.get_state_space()]
        return list(
            itertools.chain(
                *[self._state_space_n(n)
                  for n in range(1, self.max_nrepeats + 1)]))


class MultiTypeHyperparameters(HyperparameterBase):
    """TODO: This should take a list of Hyperparameters.

    Support a hyperparameter that can be multiple types.
    """
    pass


class JointHyperparameter(HyperparameterBase):
    """TODO: This should generate logically linked hyperparameters.

    Support hyperparameters that are activated only in certain combinations.
    """
    pass
