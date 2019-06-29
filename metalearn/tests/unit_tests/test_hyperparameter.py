"""Unit tests for hyperparameter module."""

import numpy as np
import sklearn

from metalearn.components import hyperparameter


def test_float_hyperparameter():
    num_hp = hyperparameter.UniformFloatHyperparameter(
        "num_hyperparam", 2, 3, default=4, log=False, n=5)
    expected = np.array([2, 2.25, 2.5, 2.75, 3, 4])
    assert (num_hp.get_state_space() == expected).all()


def test_float_log_hyperparameter():
    num_hp = hyperparameter.UniformFloatHyperparameter(
        "num_hyperparam", 1e-3, 1e-6, default=1e-7, log=True, n=4)
    expected = np.array([1e-6, 1e-5, 1e-4, 1e-3, 1e-7])
    assert (num_hp.get_state_space() == expected).all()


def test_int_hyperparameter():
    num_hp = hyperparameter.UniformIntHyperparameter(
        "num_hyperparam", 0, 100, default=5, log=False, n=5)
    expected = np.array([0, 25, 50, 75, 100, 5])
    assert (num_hp.get_state_space() == expected).all()


def test_int_log_hyperparameter():
    num_hp = hyperparameter.UniformIntHyperparameter(
        "num_hyperparam", 1, 1000, default=1000, log=True, n=4)
    expected = np.array([1, 10, 100, 1000])
    assert (num_hp.get_state_space() == expected).all()


def test_tuple_pair_hyperparameter():
    tup_hp = hyperparameter.TuplePairHyperparameter(
        "tuple_hyperparam", [
            hyperparameter.UniformIntHyperparameter(
                "hyperparam1", 1, 5, default=1, n=5),
            hyperparameter.UniformIntHyperparameter(
                "hyperparam2", 2, 6, default=2, n=5),
        ], default=(1, 2))
    expected = np.array([
        (i, j) for i in range(1, 6) for j in range(2, 7)])
    assert (tup_hp.get_state_space() == expected).all()


def test_tuple_repeating_hyperparameter():
    tup_hp = hyperparameter.TupleRepeatingHyperparameter(
        "tuple_hyperparam",
        hyperparameter.UniformIntHyperparameter(
            "hyperparam1", 1, 5, default=1, n=5),
        max_nrepeats=1, default=(1, ))
    expected = np.array([
        [(1, ), (2, ), (3, ), (4, ), (5, )]])
    assert (tup_hp.get_state_space() == expected).all()

    # max_nrepeats > 1
    mult_tup_hp = hyperparameter.TupleRepeatingHyperparameter(
        "tuple_hyperparam",
        hyperparameter.UniformIntHyperparameter(
            "hyperparam1", 1, 2, default=1, n=2),
        max_nrepeats=2, default=(1, ))
    mult_expected = np.array([
        [(1, ), (2, ), (1, 1), (1, 2), (2, 1), (2, 2)]])
    assert (mult_tup_hp.get_state_space() == mult_expected).all()


def test_base_estimator_hyperparameter():
    base_est_hp = hyperparameter.BaseEstimatorHyperparameter(
        "base_estimator_hyperparam",
        sklearn.tree.DecisionTreeClassifier,
        hyperparameters=[
            hyperparameter.UniformIntHyperparameter(
                "max_depth", 1, 10, default=1, n=10, log=False)],
        default=sklearn.tree.DecisionTreeClassifier(max_depth=1))

    expected = [
        sklearn.tree.DecisionTreeClassifier(max_depth=i + 1)
        for i in range(10)]

    for i, base_est in enumerate(base_est_hp.get_state_space()):
        assert base_est.get_params() == expected[i].get_params()


def test_embedded_estimator_hyperparameter():
    embedded_est_hp = hyperparameter.EmbeddedEstimatorHyperparameter(
        "embedded_estimator_hyperparam",
        hyperparameter.CategoricalHyperparameter(
            "hyperparam", ["a", "b", "c"], default=None))

    assert embedded_est_hp.get_state_space() == ["a", "b", "c"]
    assert embedded_est_hp.hname == "embedded_estimator_hyperparam__hyperparam"
