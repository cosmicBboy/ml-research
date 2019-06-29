"""Module for abstracting hyperparameters.

Meant to be a lighter-weight version of https://github.com/automl/ConfigSpace,
which incorporates conditional logic and more features, this API only
enumerates the state space of Hyperparameters without enforcing any rules
about how hyperparameters interact with each other. This will presumably be
the job of the Controller RNN.
"""

from collections import namedtuple
import itertools
import numpy as np

from . import constants

N_VALUES = 5  # number of values to generate for int/float state spaces


class HyperparameterBase(object):

    def __init__(self, hname, state_space, default, exclude_conditions=None):
        """Create hyperparameter base class."""
        self.hname = hname
        self._state_space = list(state_space)
        self.default = default
        self.exclude_conditions = exclude_conditions

    def __repr__(self):
        return "<%s: \"%s\">" % (type(self).__name__, self.hname)

    def default_in_state_space(self):
        return self.default in self._state_space or self.default is None

    def get_state_space(self, with_none_token=False):
        """
        :returns: list of tokens representing hyperparameter space
        :rtype: list[int|float|str]
        """
        if self.default_in_state_space():
            state_space = self._state_space
        else:
            state_space = self._state_space + [self.default]
        if with_none_token:
            state_space.append(constants.NONE_TOKEN)
        return state_space


class CategoricalHyperparameter(HyperparameterBase):

    def __init__(self, hname, state_space, default, exclude_conditions=None):
        super().__init__(hname, state_space, default, exclude_conditions)


class NumericalHyperparameter(HyperparameterBase):

    def __init__(self, hname, min, max, dtype, default, log, n,
                 exclude_conditions=None):
        self.min = dtype(min)
        self.max = dtype(max)
        self.dtype = dtype
        self.n = n
        self.log = log
        super().__init__(hname, self._init_state_space(), default,
                         exclude_conditions)

    def _init_state_space(self):
        if self.log:
            # evenly distributed in intervals in log space
            space_func = np.geomspace
        else:
            space_func = np.linspace
        return np.sort(
            space_func(self.min, self.max, self.n, dtype=self.dtype))


class UniformIntHyperparameter(NumericalHyperparameter):

    def __init__(self, hname, min, max, default, log=False, n=N_VALUES,
                 exclude_conditions=None):
        super().__init__(hname, min, max, int, default, log, n=n,
                         exclude_conditions=exclude_conditions)


class UniformFloatHyperparameter(NumericalHyperparameter):

    def __init__(self, hname, min, max, default, log=False, n=N_VALUES,
                 exclude_conditions=None):
        super().__init__(hname, min, max, float, default, log, n=n,
                         exclude_conditions=exclude_conditions)


class TuplePairHyperparameter(HyperparameterBase):
    """Tuple Pair Hyperparameter class.

    For hyperparameters in the form: `(x, y)`
    """

    def __init__(self, hname, hyperparameters, default,
                 exclude_conditions=None):
        self.hyperparameters = hyperparameters
        super().__init__(hname, self._init_state_space(), default,
                         exclude_conditions)

    def _init_state_space(self):
        return list(
            itertools.product(
                *[h.get_state_space() for h in self.hyperparameters]))


class TupleRepeatingHyperparameter(HyperparameterBase):
    """Tuple Pair Hyperparameter class.

    For hyperparameters in the form: `(x0, x1, ... , xn)`
    """

    def __init__(self, hname, hyperparameter, max_nrepeats, default,
                 exclude_conditions=None):
        self.hyperparameter = hyperparameter
        self.max_nrepeats = max_nrepeats
        super().__init__(hname, self._init_state_space(), default,
                         exclude_conditions)

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


class BaseEstimatorHyperparameter(HyperparameterBase):
    """Single Base Estimator Hyperparameter class for ensemble methods.

    For example, for AdaBoost or Bagging estimators.
    """

    def __init__(self, hname, base_estimator, hyperparameters, default,
                 exclude_conditions=None):
        """Initialize base estimator hyperparameter."""
        self.base_estimator = base_estimator
        self.hyperparameters = hyperparameters
        super().__init__(hname, self._init_state_space(), default,
                         exclude_conditions)

    def default_in_state_space(self):
        for base_est in self._state_space:
            if self.default.get_params() == base_est.get_params():
                return True
        return False

    def _init_state_space(self):
        # TODO: similar implementation of this in
        # AlgorithmComponent.hyperparameter_iterator. Consider abstracting that
        # functionality into a utils module.
        expanded_state_space = []
        for h in self.hyperparameters:
            expanded_state_space.append([
                (h.hname, v) for v in h.get_state_space()])
        return [
            self.base_estimator(**dict(hsetting)) for hsetting in
            list(itertools.product(*expanded_state_space))]


class EmbeddedEstimatorHyperparameter(HyperparameterBase):
    """Sets the hyperparameter value of an embedded estimator.

    For example, in the ColumnTransformer ``transformer`` argument.
    """

    def __init__(self, estimator_name, hyperparameter,
                 exclude_conditions=None):
        self.hyperparameter = hyperparameter
        self.estimator_name = estimator_name
        super().__init__(
            f"{self.estimator_name}__{self.hyperparameter.hname}",
            hyperparameter.get_state_space(),
            hyperparameter.default,
            exclude_conditions)


class MultiBaseEstimatorHyperparameter(HyperparameterBase):
    """Multiple Base Estimator Hyperparameter class for ensemble methods.

    TODO: this should take a list of BaseEstimatorHyperparameters and the
    state space is the concatenation of all possible estimators.
    """
    pass


class MultiTypeHyperparameters(HyperparameterBase):
    """TODO: This should take a list of Hyperparameters.

    Support a hyperparameter that can be multiple types.
    """
    pass


class ProbabalisticHyperparameterBase(object):
    """
    TODO: This should be a family of hyperparameters that are numerical and
    can be drawn from a distribution, e.g. Gaussian.
    """
    pass
