"""Module for abstracting hyperparameters.

Meant to be a lighter-weight version of https://github.com/automl/ConfigSpace,
which incorporates conditional logic and more features, this API only
enumerates the state space of Hyperparameters without enforcing any rules
about how hyperparameters interact with each other. This will presumably be
the job of the Controller RNN.
"""

import itertools
import numpy as np
from typing import Dict, Any
from ..data_types import HyperparamType

from . import constants

N_VALUES = 5  # number of values to generate for int/float state spaces


class HyperparameterBase(object):

    def __init__(self, hname, state_space=None, default=None):
        """Create hyperparameter base class."""
        self.hname = hname
        self._state_space = (
            state_space if state_space is None else list(state_space)
        )
        self.default = default

    def __repr__(self):
        return "<%s: \"%s\">" % (type(self).__name__, self.hname)

    def default_in_state_space(self):
        return self.default in self._state_space or self.default is None

    def get_state_space(self, with_none_token=False) -> Dict[str, Any]:
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
        return {
            "type": HyperparamType.CATEGORICAL,
            "choices": state_space,
        }


class CategoricalHyperparameter(HyperparameterBase):

    def __init__(self, hname, state_space, default):
        super().__init__(hname, state_space, default)


class NumericalHyperparameter(HyperparameterBase):

    def __init__(
        self, hname, min, max, dtype, default, log, n, as_categorical=False
    ):
        self.min = dtype(min)
        self.max = dtype(max)
        self.dtype = dtype
        self.n = n
        self.log = log
        self.as_categorical = as_categorical
        super().__init__(hname, self._init_state_space(), default)

    def _init_state_space(self):
        if self.log:
            # evenly distributed in intervals in log space
            space_func = np.geomspace
        else:
            space_func = np.linspace
        return np.sort(
            space_func(self.min, self.max, self.n, dtype=self.dtype))

    @property
    def type(self):
        raise NotImplementedError

    def get_state_space(
        self, with_none_token=None
    ) -> Dict[str, Any]:
        """
        :returns: dict with "min" and "max" keys
        """
        if self.as_categorical:
            return {
                "type": HyperparamType.CATEGORICAL,
                "choices": self._state_space,
            }
        return {
            "type": self.type,
            "min": self.min,
            "max": self.max,
        }


class UniformIntHyperparameter(NumericalHyperparameter):

    def __init__(
        self, hname, min, max, default=None, log=False, n=N_VALUES,
        as_categorical=False
    ):
        super().__init__(
            hname, min, max, int, default, log, n=n,
            as_categorical=as_categorical
        )

    @property
    def type(self):
        return HyperparamType.INTEGER


class UniformFloatHyperparameter(NumericalHyperparameter):

    def __init__(
        self, hname, min, max, default=None, log=False, n=N_VALUES,
        as_categorical=False
    ):
        super().__init__(
            hname, min, max, float, default, log, n=n,
            as_categorical=as_categorical
        )

    @property
    def type(self):
        return HyperparamType.REAL


class TuplePairHyperparameter(HyperparameterBase):
    """Tuple Pair Hyperparameter class.

    For hyperparameters in the form: `(x, y)`
    """

    def __init__(self, hname, hyperparameters, default):
        for hyperparam in hyperparameters:
            if (
                isinstance(hyperparam, NumericalHyperparameter)
                and not hyperparam.as_categorical
            ):
                raise ValueError(
                    "numerical hyperparameters in TuplePairHyperparameter "
                    "should be treated as categorical. Set "
                    "as_categorical=True in the constructor."
                )
        self.hyperparameters = hyperparameters
        super().__init__(hname, self._init_state_space(), default)

    def _init_state_space(self):
        return list(
            itertools.product(
                *[h.get_state_space()["choices"] for h in self.hyperparameters]
            )
        )


class EmbeddedEstimatorHyperparameter(HyperparameterBase):
    """Sets the hyperparameter value of an embedded estimator.

    For example, in the ColumnTransformer ``transformer`` argument.
    """

    def __init__(self, estimator_name, hyperparameter):
        self.hyperparameter = hyperparameter
        self.estimator_name = estimator_name
        state_space = hyperparameter.get_state_space()
        if not state_space["type"] is HyperparamType.CATEGORICAL:
            raise ValueError(
                "Only categorical hyperparameters permitted for "
                "EmbeddedEstimatorHyperparameter"
            )
        super().__init__(
            f"{self.estimator_name}__{self.hyperparameter.hname}",
            hyperparameter.get_state_space()["choices"],
            hyperparameter.default)
