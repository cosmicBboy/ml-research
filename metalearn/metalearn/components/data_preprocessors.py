"""Data Preprocessor components."""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    QuantileTransformer, OneHotEncoder, MinMaxScaler,
    StandardScaler, RobustScaler, Normalizer)
from sklearn.impute import SimpleImputer

from .algorithm_component import AlgorithmComponent
from . import constants
from .hyperparameter import (
    CategoricalHyperparameter, UniformIntHyperparameter,
    UniformFloatHyperparameter, TuplePairHyperparameter,
    BaseEstimatorHyperparameter, EmbeddedEstimatorHyperparameter)


def simple_impute_numeric():
    """Create an imputer component.

    TODO: create a ML pipeline signature for imputing numeric and categorical
          features using the ColumnTransformer transformer.
    """
    return AlgorithmComponent(
        name="NumericImputer",
        component_class=SimpleImputer,
        component_type=constants.IMPUTER,
        hyperparameters=[
            CategoricalHyperparameter(
                "strategy",
                ["mean", "median", "most_frequent", "constant"],
                default="mean"),
            CategoricalHyperparameter(
                "add_indicator", [True, False], default=True),
        ])


def simple_impute_categorical():
    """Create an imputer component.

    TODO: create a ML pipeline signature for imputing numeric and categorical
          features using the ColumnTransformer transformer.
    """
    return AlgorithmComponent(
        name="CategoricalImputer",
        component_class=SimpleImputer,
        component_type=constants.IMPUTER,
        hyperparameters=[
            CategoricalHyperparameter(
                "strategy",
                ["most_frequent"],
                default="most_frequent"),
            CategoricalHyperparameter(
                "add_indicator", [True, False], default=True),
        ])


def simple_imputer():
    """Create a categorical and numeric imputer."""
    # this is a placeholder index for which columns should be transformed is
    # task environment specific.
    PLACEHOLDER_INDEX = "<PLACEHOLDER_INDEX>"
    return AlgorithmComponent(
        name="Imputer",
        component_class=ColumnTransformer,
        component_type=constants.IMPUTER,
        hyperparameters=[
            EmbeddedEstimatorHyperparameter(
                "numeric_imputer",
                "strategy",
                ["mean", "median", "most_frequent", "constant"],
                default="mean"),
            EmbeddedEstimatorHyperparameter(
                "numeric_imputer",
                "add_indicator",
                [True, False], default=True),
            EmbeddedEstimatorHyperparameter(
                "categorical_imputer",
                "strategy",
                ["most_frequent", "constant"],
                default="mean"),
            EmbeddedEstimatorHyperparameter(
                "categorical_imputer",
                "add_indicator",
                [True, False], default=True),
        ],
        constant_hyperparameters={
            "remainder": "passthrough",
            "transformers": [
                ("numeric_imputer", SimpleImputer(), PLACEHOLDER_INDEX),
                ("categorical_imputer", SimpleImputer(), PLACEHOLDER_INDEX)
            ]}
        )


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
