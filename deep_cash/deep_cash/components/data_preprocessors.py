"""Data Preprocessor components."""

from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import (
    QuantileTransformer, OneHotEncoder, Imputer, MinMaxScaler, StandardScaler,
    RobustScaler, Normalizer
)

from .algorithm import AlgorithmComponent
from . import constants
from .hyperparameter import (
    CategoricalHyperparameter, UniformIntHyperparameter,
    UniformFloatHyperparameter, TuplePairHyperparameter)


def imputer():
    """Create an imputer component."""
    return AlgorithmComponent(
        "Imputer", Imputer, constants.IMPUTER, [
            CategoricalHyperparameter(
                "strategy", ["mean", "median", "most_frequent"],
                default="mean"),
        ])


def one_hot_encoder():
    """Create a one hot encoder component.

    DeepCASH assumes explicit handling of which features are categorical and
    which are continuous, so this component would only be applied to the
    categorical features of a particular dataset.

    Note that we defer to defaults for the following hyperparameters:
    - n_values = "auto"

    The categorical_features hyperparameter is set as a
    data-environment-dependent because the controller isn't designed to
    propose the feature columns in which to perform feature transformations.
    This may change in the future. By definition, data-environment-dependent
    hyperparameters should be supplied as an element in the `hyperparameters`
    AlgorithmComponent.__init__ argument.

    For more details, see:
    http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
    """
    return AlgorithmComponent(
        "OneHotEncoder", OneHotEncoder, constants.ONE_HOT_ENCODER, [
            CategoricalHyperparameter("sparse", [True, False], default=True)],
        env_dep_hyperparameters={"categorical_features": []})


def variance_threshold_filter():
    """Create a variance threshold filter component.

    Removes features that are below a threshold variance.
    """
    return AlgorithmComponent(
        "VarianceThresholdFilter", VarianceThreshold,
        constants.FEATURE_PREPROCESSOR, [
            UniformFloatHyperparameter("threshold", 0.0, 10.0, default=0.0)
        ])


def minmax_scaler():
    """Create a minmax scaler component."""
    return AlgorithmComponent(
        "MinMaxScaler", MinMaxScaler, constants.RESCALER)


def standard_scaler():
    """Create a standard scaler component."""
    return AlgorithmComponent(
        "StandardScaler", StandardScaler, constants.RESCALER, [
            CategoricalHyperparameter(
                "with_mean", [True, False], default=True),
            CategoricalHyperparameter(
                "with_std", [True, False], default=True),
        ])


def robust_scaler():
    """Create a robust scaler component."""
    return AlgorithmComponent(
        "RobustScaler", RobustScaler, constants.RESCALER, [
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
        "Normalizer", Normalizer, constants.RESCALER, [
            CategoricalHyperparameter(
                "norm", ["l1", "l2", "max"], default="l2")
        ])
