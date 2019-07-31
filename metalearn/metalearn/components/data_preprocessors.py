"""Data Preprocessor components."""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    QuantileTransformer, OneHotEncoder, MinMaxScaler,
    StandardScaler, RobustScaler, Normalizer)
from sklearn.impute import SimpleImputer

from .algorithm_component import AlgorithmComponent
from .hyperparameter import (
    CategoricalHyperparameter, UniformIntHyperparameter,
    UniformFloatHyperparameter, TuplePairHyperparameter,
    BaseEstimatorHyperparameter, EmbeddedEstimatorHyperparameter)
from ..data_types import AlgorithmType


def simple_imputer():
    """Create a categorical and numeric imputer."""
    # this is a placeholder index for which columns should be transformed is
    # task environment specific.

    def init_simple_imputer(
            component_class, categorical_features, numeric_features):

        return component_class(transformers=[
            ("continuous_imputer", SimpleImputer(), numeric_features),
            ("categorical_imputer", SimpleImputer(), categorical_features),
        ])

    return AlgorithmComponent(
        name="SimpleImputer",
        component_class=ColumnTransformer,
        component_type=AlgorithmType.IMPUTER,
        initialize_component=init_simple_imputer,
        hyperparameters=[
            EmbeddedEstimatorHyperparameter(
                "continuous_imputer",
                CategoricalHyperparameter(
                    "strategy", ["mean", "median"],
                    default="mean")),
            EmbeddedEstimatorHyperparameter(
                "categorical_imputer",
                CategoricalHyperparameter(
                    "strategy", ["most_frequent"],
                    default="most_frequent")),
        ],
        constant_hyperparameters={
            "remainder": "passthrough",
            "continuous_imputer__add_indicator": False,
            "categorical_imputer__add_indicator": False,
        }
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

    def init_one_hot_encoder(
            component_class, categorical_features, numeric_features):
        return component_class(transformers=[
            ("numeric_features", "passthrough", numeric_features),
            ("one_hot_encoder", OneHotEncoder(), categorical_features),
        ])

    return AlgorithmComponent(
        name="OneHotEncoder",
        component_class=ColumnTransformer,
        component_type=AlgorithmType.ONE_HOT_ENCODER,
        initialize_component=init_one_hot_encoder,
        constant_hyperparameters={
            "remainder": "passthrough",
            "one_hot_encoder__drop": None,
            "one_hot_encoder__handle_unknown": "ignore",
            "one_hot_encoder__sparse": False,
            "one_hot_encoder__categories": "auto",
        })


def minmax_scaler():
    """Create a minmax scaler component."""
    return AlgorithmComponent(
        name="MinMaxScaler",
        component_class=MinMaxScaler,
        component_type=AlgorithmType.RESCALER)


def standard_scaler():
    """Create a standard scaler component."""
    return AlgorithmComponent(
        name="StandardScaler",
        component_class=StandardScaler,
        component_type=AlgorithmType.RESCALER,
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
        component_type=AlgorithmType.RESCALER,
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
        component_type=AlgorithmType.RESCALER,
        hyperparameters=[
            CategoricalHyperparameter(
                "norm", ["l1", "l2", "max"], default="l2")
        ])
