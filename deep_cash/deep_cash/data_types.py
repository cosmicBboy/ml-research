"""Module that defines the data environment data types."""

from enum import Enum


class FeatureType(Enum):
    CATEGORICAL = 1
    CONTINUOUS = 2
    # TODO: Going to hold off on these two types of features since supporting
    # it would add more complexity to the system. Implement this when the
    # classification systems using continuous and categorical data are working.
    # STRING (for text data)
    # DATE (for time-series data)


class TargetType(Enum):
    BINARY = 1
    MULTICLASS = 2
