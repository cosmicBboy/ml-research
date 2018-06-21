"""Module for sampling from a distribution of classification environments."""

import sklearn.datasets
import sklearn.preprocessing


MULTILABEL = sklearn.preprocessing.LabelBinarizer  # for multilabel problem
MULTICLASS = "multiclass"
BINARY = "binary"


def envs():
    return [
        (sklearn.datasets.load_iris, MULTICLASS, None),
        (sklearn.datasets.load_digits, MULTICLASS, None),
        (sklearn.datasets.load_wine, MULTICLASS, None),
        (sklearn.datasets.load_breast_cancer, BINARY, None),
    ]


def env_names():
    return [n[0].__name__ for n in envs()]
