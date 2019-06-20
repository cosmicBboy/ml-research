"""Define scoring functions."""

import numpy as np
import operator
import sklearn.metrics

from collections import namedtuple
from functools import partial

# TODO: full coverage of sklearn.metrics


Scorer = namedtuple(
    "Scorer", [
        "name",
        "fn",  # sklearn-compliant scoring function e.g. mean_squared_error
        "reward_transformer",  # function that bounds range of scoring function
        # function to determine if score a is better than b. Has signature
        # x, y -> bool, returning true if x is better than y.
        "comparator",
        "needs_proba",
    ])

RegressionScorer = partial(Scorer, needs_proba=False)


def multiclass_classification_metrics():
    return [
        accuracy(),
        precision(),
        recall(),
        f1_score_weighted_average(),
        f1_score_macro(),
        log_loss(),
    ]


def binary_classification_metrics():
    return multiclass_classification_metrics() + [roc_auc()]


def regression_metrics():
    return [
        mean_absolute_error(),
        mean_squared_error(),
        r2_score(),
        root_mean_squared_error(),
        root_mean_squared_log_error(),
    ]


def exponentiated_log(x, gamma=0.1):
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
        name="accuracy",
        fn=sklearn.metrics.accuracy_score,
        reward_transformer=None,
        comparator=operator.gt,
        needs_proba=False)


def precision():
    return Scorer(
        name="precision",
        fn=sklearn.metrics.precision_score,
        reward_transformer=None,
        comparator=operator.gt,
        needs_proba=False)


def recall():
    return Scorer(
        name="recall",
        fn=sklearn.metrics.recall_score,
        reward_transformer=None,
        comparator=operator.gt,
        needs_proba=False)


def f1_score_weighted_average():
    """F1 score weighted average scorer."""
    return Scorer(
        name="f1_weighted_average",
        fn=partial(sklearn.metrics.f1_score, average="weighted"),
        reward_transformer=None,
        comparator=operator.gt,
        needs_proba=False)


def f1_score_macro():
    """F1 score macro scorer."""
    return Scorer(
        name="f1_macro",
        fn=partial(sklearn.metrics.f1_score, average="macro"),
        reward_transformer=None,
        comparator=operator.gt,
        needs_proba=False)


def log_loss():
    """Logistic loss (cross-entropy loss) scorer."""
    return Scorer(
        name="log_loss",
        fn=sklearn.metrics.log_loss,
        reward_transformer=exponentiated_log,
        comparator=operator.lt,
        needs_proba=True)


def roc_auc():
    """ROC AUC scorer."""
    return Scorer(
        name="roc_auc",
        fn=sklearn.metrics.roc_auc_score,
        reward_transformer=None,
        comparator=operator.gt,
        needs_proba=True)


# REGRESSION_METRICS

def mean_absolute_error():
    """Mean absolute error scorer."""
    return RegressionScorer(
        name="mean_absolute_error",
        fn=sklearn.metrics.mean_absolute_error,
        reward_transformer=exponentiated_log,
        comparator=operator.lt)


def mean_squared_error():
    """Mean squared error scorer."""
    return RegressionScorer(
        name="mean_squared_error",
        fn=sklearn.metrics.mean_squared_error,
        reward_transformer=exponentiated_log,
        comparator=operator.lt)


def r2_score():
    """R^2 coefficient of determination scorer."""
    return RegressionScorer(
        name="r2",
        fn=sklearn.metrics.r2_score,
        reward_transformer=rectified_linear,
        comparator=operator.gt)


def root_mean_squared_error():
    """Root mean squared error scorer."""
    def _rmse_scorer(*args, **kwargs):
        return np.sqrt(sklearn.metrics.mean_squared_error(*args, **kwargs))

    return RegressionScorer(
        name="root_mean_squared_error",
        fn=_rmse_scorer,
        reward_transformer=exponentiated_log,
        comparator=operator.lt)


def root_mean_squared_log_error():
    """Root mean squared log error."""
    def _rmsle_scorer(*args, **kwargs):
        return np.sqrt(sklearn.metrics.mean_squared_log_error(*args, **kwargs))

    return RegressionScorer(
        name="root_mean_squared_log_error",
        fn=_rmsle_scorer,
        reward_transformer=exponentiated_log,
        comparator=operator.lt)
