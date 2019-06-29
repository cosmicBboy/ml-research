import pytest

from collections import OrderedDict
from itertools import product
from metalearn.components.algorithm_component import AlgorithmComponent
from metalearn.components.constants import CLASSIFIER
from sklearn.base import BaseEstimator


class MockEstimator(BaseEstimator):

    def __init__(self, hyperparameter1=1, hyperparameter2="a",
                 categorical_features=None, continuous_features=None):
        self.hyperparameter1 = hyperparameter1
        self.hyperparameter2 = hyperparameter1
        self.categorical_features = categorical_features
        self.continuous_features = continuous_features


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
        initialize_component=None,
        hyperparameters=None,
        exclusion_conditions=None):
    return AlgorithmComponent(
        name="TestComponent",
        component_class=MockEstimator,
        component_type=component_type,
        initialize_component=initialize_component,
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
    """Test that exclusion conditions correctly render exclusion mask."""
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


def test_initialize_component():
    """Test that initialize component creates estimator with task metadata."""

    def init_component(
            component_class, categorical_features, continuous_features):
        return component_class(
            categorical_features=categorical_features,
            continuous_features=continuous_features)

    algorithm_component = _algorithm_component(
        initialize_component=init_component,
        hyperparameters=MOCK_HYPERPARAMS)

    for cat_feat, cont_feat in [
            ([1, 2, 3], [10, 11, 12]),
            ([1, 2, 3], [10, 11, 12]),
            ]:
        estimator = algorithm_component(
            categorical_features=[1, 2, 3],
            continuous_features=[10, 11, 12])
        assert estimator.categorical_features == [1, 2, 3]
        assert estimator.continuous_features == [10, 11, 12]

    # case where init_component is None
    algorithm_component = _algorithm_component(
        initialize_component=None,
        hyperparameters=MOCK_HYPERPARAMS)

    estimator = algorithm_component(
        categorical_features=[1, 2, 3],
        continuous_features=[10, 11, 12])
    assert estimator.categorical_features is None
    assert estimator.continuous_features is None


def test_initialize_component_exceptions():
    """Test function signature exceptions of init_component."""

    algorithm_component = _algorithm_component(
        initialize_component=lambda: None)

    # function signature needs to be:
    # (component: Estimator|Transformer,
    #  categorical_features: List[int],
    #  categorical_features: List[int]) -> Estimator
    for fn in [
            lambda: None,
            lambda x: None,
            lambda x, y: None,
            lambda x, y, z, _: None,
            ]:
        with pytest.raises(TypeError):
            algorithm_component = _algorithm_component(
                initialize_component=fn)
            algorithm_component()

    # Test estimator is returned
    for fn in [
            lambda component_class, cat_feats, cont_feats: component_class(
                categorical_features=cat_feats,
                continuous_features=cont_feats),
            lambda component_class, *args: component_class()
            ]:

        algorithm_component = _algorithm_component(
            initialize_component=fn)
        assert isinstance(algorithm_component(), BaseEstimator)

    # when algorithm component output is not an estimator
    for fn in [
            lambda x, y, z: None,
            lambda x, y, z: "foobar",
            lambda x, y, z: 1,
            lambda x, y, z: 0.0511235,
            lambda x, y, z: [],
            lambda x, y, z: {},
            ]:
        with pytest.raises(TypeError):
            algorithm_component = _algorithm_component(
                initialize_component=fn)
            algorithm_component()
