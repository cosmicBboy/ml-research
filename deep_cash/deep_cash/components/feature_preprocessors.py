"""Feature Preprocessor components.

NOTE:
- In order to use estimators as feature selectors, you can use
  sklearn.feature_selection.SelectFromModel
- Many of the decisions to use/exclude certain hyperparameters, as well as
  ranges for numerical hyperparameters followed the work done by the
  auto-sklearn project:

  https://github.com/automl/auto-sklearn/tree/master/autosklearn/pipeline/components

TODO:
- SelectPercentile feature selector works for classification using the
  following parameters: ["chi2", "f_classif", "mutual_info"]
  - for regression, it's ["f_regression", "mutual_info"].
"""

from sklearn.cluster import FeatureAgglomeration
from sklearn.decomposition import PCA, FastICA, KernelPCA, TruncatedSVD
from sklearn.ensemble import (
    RandomTreesEmbedding, ExtraTreesClassifier, ExtraTreesRegressor)
from sklearn.feature_selection import (
    SelectFromModel, SelectPercentile, GenericUnivariateSelect)
from sklearn.kernel_approximation import RBFSampler, Nystroem
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import LinearSVC

from .algorithm import AlgorithmComponent
from .hyperparameter import (
    CategoricalHyperparameter, UniformIntHyperparameter,
    UniformFloatHyperparameter, TuplePairHyperparameter)


# constant used for rbf kernel
# https://github.com/automl/auto-sklearn/blob/master/autosklearn/pipeline/components/feature_preprocessing/kitchen_sinks.py
GAMMA_MIN = 3.0517578125e-05


def extra_trees_classification_preprocessor():
    # TODO: This component needs to know if the supervised task in the ML
    # framework is a classification or regression task
    pass


def extra_trees_regression_preprocessor():
    # TODO: This component needs to know if the supervised task in the ML
    # framework is a classification or regression task
    pass


def fast_ica():
    """Create FastICA component.

    Leave out the following hyperparameters:
    - fun_args
    - max_iter
    - tol
    - w_init
    - random_state
    """
    return AlgorithmComponent(
        "FastICA", FastICA, [
            UniformIntHyperparameter(
                "n_components", 10, 2000, default=None, log=True),
            CategoricalHyperparameter(
                "algorithm", ["parallel", "deflation"], default="parallel"),
            CategoricalHyperparameter(
                "whiten", [True, False], default=True),
            CategoricalHyperparameter(
                "fun", ["logcosh", "exp", "cube"], default="logcosh"),
        ])


def feature_agglomeration():
    """Create FeatureAgglomoration component."""
    return AlgorithmComponent(
        "FeatureAgglomeration", FeatureAgglomeration, [
            UniformIntHyperparameter("n_clusters", 2, 400, default=2, n=20),
            CategoricalHyperparameter(
                "affinity", ["euclidean", "l1", "l2", "manhattan", "cosine"],
                default="euclidean"),
            CategoricalHyperparameter(
                "linkage", ["ward", "complete", "average"], default="ward"),
            CategoricalHyperparameter(
                "pooling_func", ["mean", "median", "max"], default="mean"),
        ])


def kernel_pca():
    """Create KernelPCA component.

    Leave out the following hyperaparameters:
    - kernel_params
    - alpha
    - fit_inverse_transform
    - eigen_solver
    - tol
    - max_iter
    - remove_zero_eig
    - random_state
    """
    return AlgorithmComponent(
        "KernelPCA", KernelPCA, [
            UniformIntHyperparameter("n_components", 10, 2000, default=100),
            CategoricalHyperparameter(
                "kernel", ["poly", "rdf", "sigmoid", "cosine"], default="rbf"),
            UniformFloatHyperparameter(
                "gamma", GAMMA_MIN, 8.0, default=1.0, log=True),
            UniformIntHyperparameter("degree", 2, 5, 3),
            UniformIntHyperparameter("coef0", -1, 1, default=0),
        ])


def rbf_sampler():
    """Create RBF Sampler 'Kitchen Sink' Component."""
    return AlgorithmComponent(
        "RBFSample", RBFSampler, [
            UniformFloatHyperparameter("gamma", GAMMA_MIN, 8.0, default=1.0),
            UniformIntHyperparameter(
                "n_components", 50, 10000, default=100, log=True, n=10),
        ])


def liblinear_svc_preprocessor():
    # TODO: This component needs to know if the supervised task in the ML
    # framework is a classification or regression task
    pass


def nystroem_sampler():
    """Create Nystroem Component."""
    return AlgorithmComponent(
        "Nystroem", Nystroem, [
            CategoricalHyperparameter(
                "kernel", ["poly", "rbf", "sigmoid", "cosine"], default="rbf"),
            UniformIntHyperparameter(
                "n_components", 50, 10000, default=100, log=True),
            UniformFloatHyperparameter(
                "gamma", GAMMA_MIN, 8.0, log=True, default=1.0),
            UniformIntHyperparameter("degree", 2, 5, 3),
            UniformFloatHyperparameter("coef0", -1, 1, default=0),
        ])


def pca():
    """Create a PCA Component.

    Note here that we use floats for the n_components TuplePairHyperparameter,
    for more details see:

    http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

    Leave out the following hyperparameters:
    - svd_solver: always use "full"
    - tol: only used when ``svd_solver == "arpack"``
    - iterated_power: only used when ``svd_solver == "randomized"``
    """
    return AlgorithmComponent(
        "PCA", PCA, [
            UniformFloatHyperparameter(
                "n_components", 0.5, 0.999, default=0.999),
            CategoricalHyperparameter("whiten", [True, False], default=False)
        ])


def polynomial_features():
    """Create Polynomial Component."""
    return AlgorithmComponent(
        "PolynomialFeatures", PolynomialFeatures, [
            UniformIntHyperparameter("degree", 2, 3, default=2, n=2),
            CategoricalHyperparameter(
                "interaction_only", [True, False], default=False),
            CategoricalHyperparameter(
                "include_bias", [True, False], default=True)
        ])


def random_trees_embedding():
    """Create RandomTreesEmbedding."""
    return AlgorithmComponent(
        "RandomTreesEmbedding", RandomTreesEmbedding, [
            UniformIntHyperparameter("n_estimators", 10, 100, default=10),
            UniformIntHyperparameter("max_depth", 2, 10, default=5),
            UniformIntHyperparameter("min_samples_split", 2, 20, default=2),
            UniformIntHyperparameter("min_samples_leaf", 1, 20, default=1),
            UniformFloatHyperparameter(
                "min_weight_fraction_leaf", 0.0, 1.0, default=1.0),
            UniformIntHyperparameter(
                "max_leaf_nodes", 10, 1000, default=None, log=True),
        ])


def select_percentile_classification():
    # TODO: This component needs to know if the supervised task in the ML
    # framework is a classification or regression task
    pass


def select_percentile_regression():
    # TODO: This component needs to know if the supervised task in the ML
    # framework is a classification or regression task
    pass


def truncated_svd():
    return AlgorithmComponent(
        "TruncatedSVD", TruncatedSVD, [
            UniformIntHyperparameter("n_components", 10, 256, default=128)
        ])
