"""Module for creating a structured algorithm environment.

The structured algorithm space is different from the `algorithm_env.py`
module because here we specify an interface to generate a machine learning
framework F, which is a sequence of algorithms (estimators/transformers) and
their associated hyperparameters. This interface is inspired by this paper
on automated machine learning:

papers.nips.cc/paper/5872-efficient-and-robust-automated-machine-learning.pdf

The sequence follows the general structure

.-------------------.    .----------------------.    .----------------------.
| data preprocessor | -> | feature preprocessor | -> | classifier/regressor |
.-------------------.    .----------------------.    .----------------------.

where there can be n data preprocessors (imputation, scaling, encoding,
filtering).
"""

import itertools
import numpy as np

from sklearn.pipeline import Pipeline

from . import components

START_TOKEN = "<sos>"
END_TOKEN = "<eos>"
NONE_TOKEN = "<none>"


class AlgorithmSpace(object):
    """A class that generates machine learning frameworks."""

    # currently support data preprocessor, feature preprocessor, classifer
    N_COMPONENT_TYPES = 3

    def __init__(self, data_preprocessors=None, feature_preprocessors=None,
                 classifiers=None, with_start_token=True,
                 with_end_token=False, with_none_token=False):
        """Initialize a structured algorithm environment.

        :param list[AlgorithmComponent]|None data_preprocessors: algorithm
            components for data preprocessing
        :param list[AlgorithmComponent]|None feature_preprocessors: algorithm
            components for feature preprocessing
        :param list[AlgorithmComponent]|None classifiers: algorithm
            components for classification
        """
        self.data_preprocessors = get_data_preprocessors() \
            if data_preprocessors is None else data_preprocessors
        self.feature_preprocessors = get_feature_preprocessors() \
            if feature_preprocessors is None else feature_preprocessors
        self.classifiers = get_classifiers() if classifiers is None \
            else classifiers
        self.with_start_token = with_start_token
        self.with_end_token = with_end_token
        self.with_none_token = with_none_token

    @property
    def components(self):
        """Concatenates all components into a single list"""
        components = self.data_preprocessors + \
            self.feature_preprocessors + \
            self.classifiers
        if self.with_start_token:
            components += [START_TOKEN]
        if self.with_end_token:
            components += [END_TOKEN]
        if self.with_none_token:
            components += [NONE_TOKEN]
        return components

    @property
    def start_token_index(self):
        return self.components.index(START_TOKEN) if self.with_start_token \
            else None

    @property
    def end_token_index(self):
        return self.components.index(END_TOKEN) if self.with_end_token \
            else None

    @property
    def none_token_index(self):
        return self.components.index(NONE_TOKEN) if self.with_none_token \
            else None

    @property
    def n_components(self):
        return len(self.components)

    def sample_ml_framework(self, random_state=None):
        """Sample a random ML framework from the algorithm space.

        :param int|None random_state: provide random state, which determines
            the ML framework sampled.
        """
        np.random.seed(random_state)
        components = [
            self.data_preprocessors[
                np.random.randint(len(self.data_preprocessors))],
            self.feature_preprocessors[
                np.random.randint(len(self.feature_preprocessors))],
            self.classifiers[
                np.random.randint(len(self.classifiers))],
        ]
        framework_hyperparameters = {}
        for a in components:
            framework_hyperparameters.update(
                a.sample_hyperparameter_state_space())
        return self._create_ml_framework(
            components, **framework_hyperparameters)

    def framework_iterator(self):
        """Return a generator of all algorithm and hyperparameter combos.

        This is potentially a huge space, creating a generator that yields
        a machine learning framework (sklearn.Pipeline object) based on all
        possible estimator combinations and all possible hyperparameter
        combinations of those estimators.
        """
        return (
            self.create_ml_framework(
                [dp, fp, clf], **self._combine_dicts([d_hp, f_hp, c_hp]))
            for dp, fp, clf in itertools.product(
                self.data_preprocessors,
                self.feature_preprocessors,
                self.classifiers)
            for d_hp, f_hp, c_hp in itertools.product(
                dp.hyperparameter_iterator(),
                fp.hyperparameter_iterator(),
                clf.hyperparameter_iterator())
        )

    def create_ml_framework(
            self, components, memory=None, **framework_hyperparameters):
        """Create ML framework, in this context an sklearn pipeline object.

        :param list[AlgorithmComponent] components: A list of algorithm
            components with which to create an ML framework.
        :param str|None memory: path to caching directory in which to store
            fitten transformers of the sklearn.Pipeline. If None, no caching
            is done
        :param dict framework_hyperparameters: hyperparameters of the pipeline.
        """
        pipeline = Pipeline(
            memory=memory,
            steps=[(a.aname, a.aclass()) for a in components])
        pipeline.set_params(**framework_hyperparameters)
        return pipeline

    def _combine_dicts(self, dicts):
        combined_dicts = {}
        for d in dicts:
            combined_dicts.update(d)
        return combined_dicts


def get_data_preprocessors():
    """Get all data preprocessors in structured algorithm environment."""
    return [
        components.data_preprocessors.imputer(),
        components.data_preprocessors.one_hot_encoder(),
        components.data_preprocessors.variance_threshold_filter(),
        components.data_preprocessors.minmax_scaler(),
        components.data_preprocessors.standard_scaler(),
        components.data_preprocessors.robust_scaler(),
        components.data_preprocessors.normalizer(),
    ]


def get_feature_preprocessors():
    """Get all feature preprocessors in structured algorithm environment."""
    return [
        components.feature_preprocessors.fast_ica(),
        components.feature_preprocessors.feature_agglomeration(),
        components.feature_preprocessors.kernel_pca(),
        components.feature_preprocessors.rbf_sampler(),
        components.feature_preprocessors.nystroem_sampler(),
        components.feature_preprocessors.pca(),
        components.feature_preprocessors.polynomial_features(),
        components.feature_preprocessors.random_trees_embedding(),
        components.feature_preprocessors.truncated_svd(),
    ]


def get_classifiers():
    """Get all classifiers in structured algorithm environment."""
    return [
        components.classifiers.logistic_regression(),
        components.classifiers.gaussian_naive_bayes(),
        components.classifiers.decision_tree(),
    ]
