"""Unit tests for hyperparameter module."""

import numpy as np
import sklearn

from metalearn.components import hyperparameter
from metalearn.data_types import HyperparamType


def test_float_hyperparameter():
    num_hp = hyperparameter.UniformFloatHyperparameter(
        "num_hyperparam", 2, 3, default=4, log=False, n=5)
    state_space = num_hp.get_state_space()
    assert state_space["type"] is HyperparamType.REAL
    assert state_space["min"] == 2
    assert state_space["max"] == 3


def test_float_log_hyperparameter():
    num_hp = hyperparameter.UniformFloatHyperparameter(
        "num_hyperparam", 1e-3, 1e-6, default=1e-7, log=True, n=4)
    state_space = num_hp.get_state_space()
    assert state_space["type"] is HyperparamType.REAL
    assert state_space["min"] == 1e-3
    assert state_space["max"] == 1e-6


def test_int_hyperparameter():
    num_hp = hyperparameter.UniformIntHyperparameter(
        "num_hyperparam", 0, 100, default=5, log=False, n=5)
    state_space = num_hp.get_state_space()
    assert state_space["type"] is HyperparamType.INTEGER
    assert state_space["min"] == 0
    assert state_space["max"] == 100


def test_int_log_hyperparameter():
    num_hp = hyperparameter.UniformIntHyperparameter(
        "num_hyperparam", 1, 1000, default=1000, log=True, n=4)
    state_space = num_hp.get_state_space()
    assert state_space["type"] is HyperparamType.INTEGER
    assert state_space["min"] == 1
    assert state_space["max"] == 1000


def test_tuple_pair_hyperparameter():
    tup_hp = hyperparameter.TuplePairHyperparameter(
        "tuple_hyperparam", [
            hyperparameter.UniformIntHyperparameter(
                "hyperparam1", 1, 5, default=1, n=5, as_categorical=True
            ),
            hyperparameter.UniformIntHyperparameter(
                "hyperparam2", 2, 6, default=2, n=5, as_categorical=True
            ),
        ], default=(1, 2))
    expected = np.array([
        (i, j) for i in range(1, 6) for j in range(2, 7)])
    assert (tup_hp.get_state_space()["choices"] == expected).all()


def test_embedded_estimator_hyperparameter():
    embedded_est_hp = hyperparameter.EmbeddedEstimatorHyperparameter(
        "embedded_estimator_hyperparam",
        hyperparameter.CategoricalHyperparameter(
            "hyperparam", ["a", "b", "c"], default=None))

    assert embedded_est_hp.get_state_space()["choices"] == ["a", "b", "c"]
    assert embedded_est_hp.hname == "embedded_estimator_hyperparam__hyperparam"
