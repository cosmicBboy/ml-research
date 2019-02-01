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


def default_logger(tracker, prefix=""):

    metrics = ["losses", "mean_validation_scores", "mean_rewards",
               "aggregate_gradients", "mlf_diversity"]
    if prefix:
        display_names = ["%s__%s" % (prefix, m) for m in metrics]
    else:
        display_names = metrics
    msg = "\n"
    for n, m in zip(display_names, metrics):
        msg += "%s: %0.02f - " % (n, tracker.history[m][-1])
    print(msg)


def floyd_logger(tracker, prefix=""):
    """Log metrics for floydhub.

    prints metrics to stdout as required by floydhub metrics tracking:
    https://docs.floydhub.com/guides/jobs/metrics/

    :param MetricsTracker tracker: tracker object for deep CASH.
    :param str prefix: prefix metric names with this
    """
    metrics = _metrics()
    metrics_dict = {}
    if prefix:
        metrics_dict = {m: "%s__%s" % (prefix, m) for m in metrics}
    for metric in metrics:
        value = tracker.history[metric][-1]
        step = tracker.history["episode"][-1]
        if value is None or np.isnan(value):
            continue
        _log_floyd(metrics_dict.get(metric, metric), value, step)


def floyd_multiprocess_logger(
        tracker, prefix="", metric="mean_rewards"):
    """Log metrics for floydhub in the case of multiprocessing jobs.

    logs mean_rewards to indicate overall progress of performance.

    :param MetricsTracker tracker: tracker object for deep CASH.
    :param str prefix: prefix metric names with this
    :param str metric: metric to display
    """
    metric_name = prefix + metric if prefix else metric
    if tracker.history[metric] > 0:
        _log_floyd(
            metric_name,
            tracker.history[metric][-1],
            tracker.history["episode"][-1])


def get_loggers():
    """Get dictionary of loggers."""
    return {
        "empty": empty_logger,
        "default": default_logger,
        "floyd": floyd_logger,
        "floyd_multiprocess": floyd_multiprocess_logger,
    }
