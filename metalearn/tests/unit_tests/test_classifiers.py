"""Unit tests for classifier algorithm components."""

import sklearn

from metalearn.components import classifiers
from sklearn.base import BaseEstimator


CLASSIFIERS = [
    classifiers.adaboost,
    classifiers.decision_tree,
    classifiers.gaussian_naive_bayes,
    classifiers.gradient_boosting,
    classifiers.multinomial_naive_bayes,
    classifiers.k_nearest_neighbors,
    classifiers.logistic_regression,
    classifiers.rbf_gaussian_process_classifier,
    classifiers.random_forest_classifier,
    classifiers.support_vector_classifier_linear,
    classifiers.support_vector_classifier_nonlinear,
]


def test_classifier_components():
    """Ensure that a classifier can be fitted and used to make predictions."""
    iris = sklearn.datasets.load_iris()
    for algorithm_component in CLASSIFIERS:
        clf_component = algorithm_component()

        hyperparam_name_space = clf_component.hyperparameter_name_space()
        if hyperparam_name_space is None:
            continue
        for hname in hyperparam_name_space:
            assert hname.startswith(clf_component.name)

        clf = clf_component()
        assert isinstance(clf, BaseEstimator)
        clf.fit(iris.data, iris.target)
        y_hat = clf.predict(iris.data)
        acc = sklearn.metrics.accuracy_score(iris.target, y_hat)
        assert 0 <= acc <= 1


def test_classifier_set_params():

    for algorithm_component in CLASSIFIERS:
        clf_component = algorithm_component()
        estimator = clf_component()
        for _ in range(20):
            hyperparams = {
                k.split("__")[1]: v for k, v in
                clf_component.sample_hyperparameter_state_space().items()
            }
            assert isinstance(
                estimator.set_params(**hyperparams), BaseEstimator)
