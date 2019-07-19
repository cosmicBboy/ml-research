"""Module that defines the data environment data types."""

from enum import Enum


class AlgorithmType(Enum):

    ONE_HOT_ENCODER = "one_hot_encoder"
    IMPUTER = "imputer"
    RESCALER = "rescaler"
    FEATURE_PREPROCESSOR = "feature_preprocessor"
    CLASSIFIER = "classifier"
    REGRESSOR = "regressor"


class FeatureType(Enum):
    """Feature type definition, effects the MLF proposal path."""

    CATEGORICAL = 1
    CONTINUOUS = 2
    DATE = 3
    # TODO: Going to hold off on these two types of features since supporting
    # it would add more complexity to the system. Implement this when the
    # classification systems using continuous and categorical data are working.
    # STRING (for text data)


class TargetType(Enum):
    """Target type definition, effects MLF proposal path."""

    BINARY = 1
    MULTICLASS = 2
    REGRESSION = 3
    MULTIREGRESSION = 4


class OpenMLTaskType(Enum):
    """Define task type and ids according to openml docs.

    https://openml.github.io/openml-python/dev/generated/openml.tasks.list_tasks.html#openml.tasks.list_tasks
    """

    # Note: only using SUPERVISED_REGRESSION at the moment
    SUPERVISED_CLASSIFICATION = 1
    SUPERVISED_REGRESSION = 2
    LEARNING_CURVE = 3
    SUPERVISED_DATA_STREAM_CLASSIFICATION = 4
    CLUSTERING = 5
    MACHINE_LEARNING_CHALLENGE = 6
    SURVIVAL_ANALYSIS = 7
    SUBGROUP_DISCOVERY = 8


class DataSourceType(Enum):
    """Define a type of data sources."""

    SKLEARN = 1
    OPEN_ML = 2
    KAGGLE = 3
    AUTOSKLEARN_BENCHMARK = 4
    KIOSK = 5


class CASHComponent(Enum):
    """A component that the CASH controller predicts to form MLF."""

    ALGORITHM = 1
    HYPERPARAMETER = 2


class ExperimentType(Enum):
    """Experiment types."""

    METALEARN_REINFORCE = 1
    RANDOM_SEARCH = 2
