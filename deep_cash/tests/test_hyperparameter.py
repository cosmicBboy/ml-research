"""Unit tests for hyperparameter module."""

import numpy as np

from deep_cash.components import hyperparameter


def test_float_hyperparameter():
    num_hp = hyperparameter.UniformFloatHyperparameter(
        "num_hyperparam", 2, 3, default=4, log=False, n=5)
    expected = np.array([2, 2.25, 2.5, 2.75, 3, 4])
    assert (num_hp.state_space == expected).all()


def test_float_log_hyperparameter():
    num_hp = hyperparameter.UniformFloatHyperparameter(
        "num_hyperparam", 1e-3, 1e-6, default=1e-7, log=True, n=4)
    expected = np.array([1e-7, 1e-6, 1e-5, 1e-4, 1e-3])
    assert (num_hp.state_space == expected).all()


def test_int_hyperparameter():
    num_hp = hyperparameter.UniformIntHyperparameter(
        "num_hyperparam", 0, 100, default=5, log=False, n=5)
    expected = np.array([0, 5, 25, 50, 75, 100])
    assert (num_hp.state_space == expected).all()


def test_int_log_hyperparameter():
    num_hp = hyperparameter.UniformIntHyperparameter(
        "num_hyperparam", 1, 1000, default=1000, log=True, n=4)
    expected = np.array([1, 10, 100, 1000])
    assert (num_hp.state_space == expected).all()
