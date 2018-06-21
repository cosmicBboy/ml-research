"""Logger module."""

import numpy as np


METRICS = [
    "losses",
    "mean_rewards",
    "mean_validation_scores",
    "std_validation_scores",
    "n_successful_mlfs",
    "n_unique_mlfs",
    "mlf_framework_diversity",
    "best_validation_scores",
]


def floyd_logger(cash_reinforce):
    """Log metrics for floydhub.

    prints metrics to stdout as required by floydhub metrics tracking:
    https://docs.floydhub.com/guides/jobs/metrics/

    :param CASHReinforce cash_reinforce: reinforce object for deep CASH.
    """
    for metric in METRICS:
        value = getattr(cash_reinforce, metric)[-1]
        if value is None or np.isnan(value):
            continue
        print('{"metric": "%s", "value": %0.10f}' % (metric, value))


def get_loggers():
    """Get dictionary of loggers."""
    return {
        "floyd": floyd_logger,
    }
