"""Module for sampling from a distribution of classification environments."""

import sklearn.datasets
import sklearn.preprocessing


MULTILABEL = sklearn.preprocessing.LabelBinarizer  # for multilabel problem
MULTICLASS = "multiclass"
BINARY = "binary"


def envs():
    return [
        ("iris", sklearn.datasets.load_iris, MULTICLASS, None),
        ("digits", sklearn.datasets.load_digits, MULTICLASS, None),
        ("wine", sklearn.datasets.load_wine, MULTICLASS, None),
        ("breast_cancer", sklearn.datasets.load_breast_cancer, BINARY, None),
    ]


def env_names():
    return [n[0] for n in envs()]
