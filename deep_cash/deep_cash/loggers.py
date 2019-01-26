"""Logger module."""

import numpy as np


def _metrics():
    return [
        "losses",
        "aggregate_gradients",
        "mean_rewards",
        "mean_validation_scores",
        "std_validation_scores",
        "n_successful_mlfs",
        "n_unique_mlfs",
        "mlf_diversity",
        "n_unique_hyperparams",
        "hyperparam_diversity",
        "best_validation_scores",
    ]


def empty_logger(*args, **kwargs):
    pass


def _log_floyd(metric, value, step):
    print('{"metric": "%s", "value": %0.10f, "step": %d}' %
          (metric, value, step))


def default_logger(tracker):
    print(
        "\nloss: %0.02f - "
        "mean performance: %0.02f - "
        "mean reward: %0.02f - "
        "grad agg: %0.02f - "
        "mlf diversity: %d/%d" % (
            tracker["losses"][-1],
            tracker["mean_validation_scores"][-1],
            tracker["mean_rewards"][-1],
            tracker["aggregate_gradients"][-1],
            tracker["mlf_diversity"][-1],
        ),
    )


def floyd_logger(tracker, prefix=""):
    """Log metrics for floydhub.

    prints metrics to stdout as required by floydhub metrics tracking:
    https://docs.floydhub.com/guides/jobs/metrics/

    :param CASHReinforce cash_reinforce: reinforce object for deep CASH.
    :param str prefix: prefix metric names with this
    """
    metrics = _metrics()
    metrics_dict = {}
    if prefix:
        metrics_dict = {m: "%s__%s" % (prefix, m) for m in metrics}
    for metric in metrics:
        value = tracker[metric][-1]
        step = tracker["episodes"][-1]
        if value is None or np.isnan(value):
            continue
        _log_floyd(metrics_dict.get(metric, metric), value, step)


def floyd_multiprocess_logger(
        cash_reinforce, prefix="", metric="mean_rewards"):
    """Log metrics for floydhub in the case of multiprocessing jobs.

    logs mean_rewards to indicate overall progress of performance.

    :param CASHReinforce cash_reinforce: reinforce object for deep CASH.
    :param str prefix: prefix metric names with this
    :param str metric: metric to display
    """
    metric_name = prefix + metric if prefix else metric
    _log_floyd(
        metric_name,
        getattr(cash_reinforce, metric)[-1],
        getattr(cash_reinforce, "episodes")[-1])


def get_loggers():
    """Get dictionary of loggers."""
    return {
        "floyd": floyd_logger,
        "floyd_multiprocess": floyd_multiprocess_logger,
    }
