"""Define scoring functions."""

import numpy as np
import operator
import sklearn.metrics

from collections import namedtuple
from functools import partial

# TODO: full coverage of sklearn.metrics


Scorer = namedtuple(
    "Scorer", [
        "fn",  # sklearn-compliant scoring function e.g. mean_squared_error
        "reward_transformer",  # function that bounds range of scoring function
        # function to determine if score a is better than b. Has signature
        # x, y -> bool, returning try if x is better than y.
        "comparator",
    ])


def exponentiated_log(x, gamma=0.01):
    """Bounds functions with a range of >= 0 to range of [0, 1].

    This function is used for scorers where larger scores mean worse
    performance. This function returns 1.0 if the score is 0, and returns
    values that asymptotically approach 0 as the score approaches positive
    infinity.

    :param float x: value to transform
    :param float gamma: decay coefficient. Larger gamma means function decays
        more quickly as x gets larger.
    :returns: transformed value.
    :rtype: float
    """
    if x < 0:
        raise ValueError("value %s not a valid input. Must be >= 0")
    if x == 0:
        # since the below function is undefined at x=0, return 1 if x=0.
        return 1.0
    return 1 / (1 + np.power(np.e, np.log(gamma * x)))


def rectified_linear(x):
    """Bounds functions with a range of [-infinity, 1] to [0, 1].

    Negative values are clipped to 0.

    This function is used for scorers, e.g. R^2 (regression metric), where
    1 is the best possible score, 0 indicates expected value of y, and negative
    numbers are worse than predicting the mean.
    """
    return 0 if x < 0 else x


# CLASSIFICATION METRICS

def accuracy():
    """Accuracy scorer."""
    return Scorer(
        fn=sklearn.metrics.accuracy_score,
        reward_transformer=None,
        comparator=operator.gt)


def f1_score_weighted_average():
    """F1 score weighted average scorer."""
    return Scorer(
        fn=partial(sklearn.metrics.f1_score, average="weighted"),
        reward_transformer=None,
        comparator=operator.gt)


def f1_score_macro():
    """F1 score macro scorer."""
    return Scorer(
        fn=partial(sklearn.metrics.f1_score, average="macro"),
        reward_transformer=None,
        comparator=operator.gt)


def log_loss():
    """Logistic loss (cross-entropy loss) scorer."""
    return Scorer(
        fn=sklearn.metrics.log_loss,
        reward_transformer=exponentiated_log,
        comparator=operator.lt)


def roc_auc():
    """ROC AUC scorer."""
    return Scorer(
        fn=sklearn.metrics.roc_auc_score,
        reward_transformer=None,
        comparator=operator.gt)


# REGRESSION_METRICS

def mean_absolute_error():
    """Mean absolute error scorer."""
    return Scorer(
        fn=sklearn.metrics.mean_absolute_error,
        reward_transformer=exponentiated_log,
        comparator=operator.lt)


def mean_squared_error():
    """Mean squared error scorer."""
    return Scorer(
        fn=sklearn.metrics.mean_squared_error,
        reward_transformer=exponentiated_log,
        comparator=operator.lt)


def r2_score():
    """R^2 coefficient of determination scorer."""
    return Scorer(
        fn=sklearn.metrics.r2_score,
        reward_transformer=rectified_linear,
        comparator=operator.gt)


def root_mean_squared_error():
    """Root mean squared error scorer."""
    def _rmse_scorer(*args, **kwargs):
        return np.sqrt(sklearn.metrics.mean_squared_error(*args, **kwargs))

    return Scorer(
        fn=_rmse_scorer,
        reward_transformer=exponentiated_log,
        comparator=operator.lt)


def root_mean_squared_log_error():
    """Root mean squared log error."""
    def _rmsle_scorer(*args, **kwargs):
        return np.sqrt(sklearn.metrics.mean_squared_log_error(*args, **kwargs))

    return Scorer(
        fn=_rmsle_scorer,
        reward_transformer=exponentiated_log,
        comparator=operator.lt)