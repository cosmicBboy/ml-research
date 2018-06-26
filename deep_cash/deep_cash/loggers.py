"""Logger module."""

import numpy as np


def _metrics():
    return [
        "losses",
        "mean_rewards",
        "mean_validation_scores",
        "std_validation_scores",
        "n_successful_mlfs",
        "n_unique_mlfs",
        "mlf_framework_diversity",
        "best_validation_scores",
    ]


def empty_logger(*args, **kwargs):
    pass


def floyd_logger(cash_reinforce, prefix=""):
    """Log metrics for floydhub.

    prints metrics to stdout as required by floydhub metrics tracking:
    https://docs.floydhub.com/guides/jobs/metrics/

    :param CASHReinforce cash_reinforce: reinforce object for deep CASH.
    """
    metrics = _metrics()
    metrics_dict = {}
    if prefix:
        metrics_dict = {m: "%s__%s" % (prefix, m) for m in metrics}
    for metric in metrics:
        value = getattr(cash_reinforce, metric)[-1]
        if value is None or np.isnan(value):
            continue
        print('{"metric": "%s", "value": %0.10f}' % (
            metrics_dict.get(metric, metric), value))


def get_loggers():
    """Get dictionary of loggers."""
    return {
        "floyd": floyd_logger,
    }
