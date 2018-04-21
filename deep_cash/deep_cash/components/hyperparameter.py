"""Module for abstracting hyperparameters."""

import numpy as np

N_VALUES = 5  # number of values to generate for int/float state spaces


class Hyperparameter(object):

    def __init__(self, hname, state_space, default):
        if default not in state_space:
            raise ValueError(
                "default %s not in state_space: %s" % (default, state_space))
        self.hname = hname
        self.state_space = state_space
        self.default = default

    def __repr__(self):
        return "<%s: \"%s\">" % (type(self).__name__, self.hname)


class CategoricalHyperparameter(Hyperparameter):

    def __init__(self, hname, state_space, default):
        super().__init__(hname, state_space, default)


class NumericalHyperparameter(Hyperparameter):

    def __init__(self, hname, min, max, dtype, default, log, n):
        self.min = min
        self.max = max
        self.dtype = dtype
        self.n = n
        self.log = log
        state_space = self.get_state_space()
        if default not in state_space:
            state_space = np.sort(
                np.concatenate([state_space,np.array([default])]))
        super().__init__(hname, state_space, default)

    def get_state_space(self):
        if self.log:
            # evenly distributed in intervals in log space
            space_func = np.geomspace
        else:
            space_func = np.linspace
        return space_func(self.min, self.max, self.n, dtype=self.dtype)


class UniformIntHyperparameter(NumericalHyperparameter):

    def __init__(self, hname, min, max, default, log=False, n=N_VALUES):
        super().__init__(hname, min, max, int, default, log, n=n)


class UniformFloatHyperparameter(NumericalHyperparameter):

    def __init__(self, hname, min, max, default, log=False, n=N_VALUES):
        super().__init__(hname, min, max, float, default, log, n=n)
