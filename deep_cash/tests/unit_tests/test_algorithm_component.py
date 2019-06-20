import pytest

from collections import OrderedDict
from itertools import product
from metalearn.components.algorithm_component import AlgorithmComponent
from metalearn.components.constants import CLASSIFIER


class MockEstimator(object):

    def __init__(self, hyperparameter1=1, hyperparameter2="a"):
        self.hyperparameter1 = hyperparameter1
        self.hyperparameter2 = hyperparameter1


class MockHyperparameter(object):

    def __init__(self, name, state_space):
        self.hname = name
        self.state_space = state_space

    def get_state_space(self, *args, **kwargs):
        return self.state_space


MOCK_HYPERPARAMS = [
    MockHyperparameter("hyperparameter1", [1, 2, 3]),
    MockHyperparameter("hyperparameter2", ["a", "b", "c"])
]


def _algorithm_component(
        component_type=CLASSIFIER,
        hyperparameters=None,
        exclusion_conditions=None):
    return AlgorithmComponent(
        name="TestComponent",
        component_class=MockEstimator,
        component_type=component_type,
        hyperparameters=hyperparameters,
        exclusion_conditions=exclusion_conditions)


def test_init():
    """Test happy path initialization and error case."""
    algorithm_component = _algorithm_component()
    assert isinstance(algorithm_component, AlgorithmComponent)

    with pytest.raises(ValueError):
        _algorithm_component("FOOBAR")


def test_call():
    """Test __call__ method correctly instantiates the algorithm object."""
    algorithm_component = _algorithm_component()
    algorithm_obj = algorithm_component()
    assert isinstance(algorithm_obj, MockEstimator)
    assert hasattr(algorithm_obj, "hyperparameter1")
    assert hasattr(algorithm_obj, "hyperparameter2")


def test_hyperparameter_name_space():
    """Test name space is correctly formatted."""
    algorithm_component = _algorithm_component(
        hyperparameters=MOCK_HYPERPARAMS)
    expected = [
        "TestComponent__hyperparameter1",
        "TestComponent__hyperparameter2"]
    result = algorithm_component.hyperparameter_name_space()
    assert result == expected

    algorithm_component_none = _algorithm_component(hyperparameters=None)
    assert algorithm_component_none.hyperparameter_name_space() is None


def test_hyperparameter_state_space():
    """Test hyperparameter state space contains correct hyperparameters."""
    algorithm_component = _algorithm_component(
        hyperparameters=MOCK_HYPERPARAMS)
    state_space = algorithm_component.hyperparameter_state_space()
    assert state_space["TestComponent__hyperparameter1"] == [1, 2, 3]
    assert state_space["TestComponent__hyperparameter2"] == ["a", "b", "c"]
    assert _algorithm_component(
        hyperparameters=None).hyperparameter_state_space() == OrderedDict()


def test_hyperparameter_iterator():
    """Test that hyperparameter iterator returns all possible combinations."""
    algorithm_component = _algorithm_component(
        hyperparameters=MOCK_HYPERPARAMS)
    hyperparam_settings = list(algorithm_component.hyperparameter_iterator())
    hnames = [
        "TestComponent__hyperparameter1",
        "TestComponent__hyperparameter2"]
    for settings in (dict(zip(hnames, s)) for s in
                     product([1, 2, 3], ["a", "b", "c"])):
        assert settings in hyperparam_settings


def test_hyperparameter_exclusion_conditions():
    """Test that exclusion conditions ."""
    algorithm_component = _algorithm_component(
        hyperparameters=MOCK_HYPERPARAMS,
        # if `1` is chosen on hyperparameter1, then exclude values "b", and "c"
        # on hyperparameter2
        exclusion_conditions={
            "hyperparameter1": {1: {"hyperparameter2": ["b", "c"]}}})
    expected = OrderedDict([
        ("TestComponent__hyperparameter1", {
            1: {"TestComponent__hyperparameter2": ["b", "c"]}})
    ])
    assert algorithm_component.hyperparameter_exclusion_conditions() == \
        expected
