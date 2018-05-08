"""Module for sampling from a distribution of classification environments."""

import sklearn.datasets
import sklearn.preprocessing


MULTICLASS_PREPROCESSOR = sklearn.preprocessing.LabelBinarizer
MULTICLASS = "multiclass"
BINARY = "binary"


def envs():
    return [
        (sklearn.datasets.load_iris, MULTICLASS, MULTICLASS_PREPROCESSOR),
        (sklearn.datasets.load_digits, MULTICLASS, MULTICLASS_PREPROCESSOR),
        (sklearn.datasets.load_wine, MULTICLASS, MULTICLASS_PREPROCESSOR),
        (sklearn.datasets.load_breast_cancer, BINARY, None),
    ]
