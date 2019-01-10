"""Data Preprocessor components."""

from sklearn.preprocessing import (
    QuantileTransformer, OneHotEncoder, Imputer, MinMaxScaler, StandardScaler,
    RobustScaler, Normalizer)

from .algorithm_component import AlgorithmComponent
from . import constants
from .hyperparameter import (
    CategoricalHyperparameter, UniformIntHyperparameter,
    UniformFloatHyperparameter, TuplePairHyperparameter)


def impute_numeric():
    """Create an imputer component.

    TODO: when this project gets to processing datasets with missing values,
    need to create another imputer function the explicitly handles numerical
    and categorical data. This will involve also modifying the
    ML_FRAMEWORK_SIGNATURE in algorithm_space.py such that there are two types
    of imputers. Will also probably need to position the OneHotEncoder
    component after the imputers.
    """
    return AlgorithmComponent(
        name="NumericImputer",
        component_class=Imputer,
        component_type=constants.IMPUTER,
        hyperparameters=[
            CategoricalHyperparameter(
                "strategy", ["mean", "median"], default="mean"),
        ])


def one_hot_encoder():
    """Create a one hot encoder component.

    DeepCASH assumes explicit handling of which features are categorical and
    which are continuous, so this component would only be applied to the
    categorical features of a particular dataset.

    The categorical_features hyperparameter is set as
    data-environment-dependent because the controller isn't designed to
    propose the feature columns in which to perform feature transformations.
    This may change in the future. By definition, data-environment-dependent
    hyperparameters should be supplied as an element in the `hyperparameters`
    AlgorithmComponent.__init__ argument.

    For more details, see:
    http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
    """
    return AlgorithmComponent(
        name="OneHotEncoder",
        component_class=OneHotEncoder,
        component_type=constants.ONE_HOT_ENCODER,
        hyperparameters=[
            CategoricalHyperparameter("sparse", [True, False], default=True)],
        env_dep_hyperparameters={
            "categorical_features": [],
            "sparse": False})


def minmax_scaler():
    """Create a minmax scaler component."""
    return AlgorithmComponent(
        name="MinMaxScaler",
        component_class=MinMaxScaler,
        component_type=constants.RESCALER)


def standard_scaler():
    """Create a standard scaler component."""
    return AlgorithmComponent(
        name="StandardScaler",
        component_class=StandardScaler,
        component_type=constants.RESCALER,
        hyperparameters=[
            CategoricalHyperparameter(
                "with_mean", [True, False], default=True),
            CategoricalHyperparameter(
                "with_std", [True, False], default=True),
        ])


def robust_scaler():
    """Create a robust scaler component."""
    return AlgorithmComponent(
        name="RobustScaler",
        component_class=RobustScaler,
        component_type=constants.RESCALER,
        hyperparameters=[
            CategoricalHyperparameter(
                "with_centering", [True, False], default=True),
            CategoricalHyperparameter(
                "with_scaling", [True, False], default=True),
            TuplePairHyperparameter(
                "quantile_range", [
                    UniformFloatHyperparameter(
                        "q_min", 0.001, 0.3, default=0.001),
                    UniformFloatHyperparameter(
                        "q_max", 0.7, 0.999, default=0.7)
                ], default=(0.25, 0.75))
        ])


def normalizer():
    """Create a normalizer component."""
    return AlgorithmComponent(
        name="Normalizer",
        component_class=Normalizer,
        component_type=constants.RESCALER,
        hyperparameters=[
            CategoricalHyperparameter(
                "norm", ["l1", "l2", "max"], default="l2")
        ])
