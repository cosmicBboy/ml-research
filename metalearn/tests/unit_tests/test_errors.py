"""Unit tests for errors module."""

import numpy as np
import sklearn
import warnings

from sklearn.exceptions import ConvergenceWarning

from metalearn.components import classifiers
from metalearn import errors


def _fit_mlf(mlf, X, y, raise_unknown_error=False):
    if raise_unknown_error:
        raise ValueError("This is an unknown fit error!")
    mlf.fit(X, y)


def test_is_valid_fit_error():
    """Test that valid fit errors correctly evaluate to True.

    This isn't a great unit test, strictly speaking, since it's testing both
    the error module and using the AlgorithmComponent module to generate
    hyperparameters.
    """
    iris = sklearn.datasets.load_iris()
    # get one observation of each label
    idx = []
    for label in np.unique(iris.target):
        idx.append(np.where(iris.target == label)[0][0])
    X = iris.data[idx]
    y = iris.target[idx]

    # ignore convergence warnings since we're only fitting on three
    # observations
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    # Here we're only testing naive bayes and logistic regression. This is
    # because we're iterating through the entire hyperparameter space.
    # TODO: support a way of reducing the search space of hyperparameters
    # so that all estimators can be quickly tested.
    for algorithm_component in [
            classifiers.gaussian_naive_bayes,
            classifiers.logistic_regression]:
        clf_component = algorithm_component()
        clf = sklearn.pipeline.Pipeline(
            [(clf_component.name, clf_component())])
        for hyperparams in clf_component.hyperparameter_iterator():
            for raise_unknown_error in [True, False]:
                clf.set_params(**hyperparams)
                try:
                    _fit_mlf(clf, X, y, raise_unknown_error)
                except Exception as error:
                    if raise_unknown_error:
                        assert not errors.is_valid_fit_error(error)
                    else:
                        import ipdb; ipdb.set_trace()
                        assert errors.is_valid_fit_error(error)
