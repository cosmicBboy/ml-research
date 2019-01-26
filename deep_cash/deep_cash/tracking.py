"""Track data during hyperparameter search."""

from collections import defaultdict


class TrackerBase(object):

    def __init__(self, metric_names):
        self.metric_names = metric_names
        self.history = defaultdict(list)

    def update_metric(self, key: str, value):
        if key not in self.metric_names:
            raise ValueError("metric %s not recognized. Must be one of: {%s}" %
                             (key, self.metric_names))
        self.history[key].append(value)

    def update_metrics(self, metrics_dict: dict):
        for key, value in metrics_dict.items():
            self.update_metric(key, value)


class MetricsTracker(TrackerBase):

    def __init__(self):
        super(MetricsTracker, self).__init__([
            "episode",
            "data_env_names",
            "scorers",
            "losses",
            "mean_rewards",
            "aggregate_gradients",
            "mean_validation_scores",
            "std_validation_scores",
            "best_validation_scores",
            "best_mlfs",
            "n_successful_mlfs",
            "n_unique_mlfs",
            "n_unique_hyperparams",
            "mlf_diversity",
            "hyperparam_diversity"
        ])
