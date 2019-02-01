"""Modules implementing random search."""

import numpy as np

from . import utils, loggers
from .algorithm_space import TARGET_TYPE_TO_MLF_SIGNATURE
from .tracking import MetricsTracker


class CASHRandomSearch(object):

    def __init__(self, a_space, t_env, metrics_logger=loggers.default_logger):
        """Initialize random search ML framework fitter."""
        self.a_space = a_space
        self.t_env = t_env
        self._metrics_logger = metrics_logger

    def fit(self, n_episodes: int, n_iter: int, verbose: bool = False,
            procnum: int = None):
        print("Running Random search CASH procedure.")
        self.tracker = MetricsTracker()
        for i_episode in range(n_episodes):
            # TODO: not sure if this for loop should be parallelized
            self.t_env.sample_data_env()
            msg = "episode %d, task: %s" % (
                i_episode, self.t_env.current_data_env.name)
            if procnum is not None:
                msg = "proc num: %d, %s" % (procnum, msg)
            print("\n" + msg)
            self.end_episode(
                i_episode, n_iter, *self.start_episode(n_iter, verbose))
            if self._metrics_logger is not None:
                self._metrics_logger(self.tracker)

    def start_episode(self, n_iter, verbose):
        mlf_signature = TARGET_TYPE_TO_MLF_SIGNATURE[
            self.t_env.current_data_env.target_type]
        mlfs, rewards, scores = [], [], []
        for i in range(n_iter):
            # TODO: this for loop can be parallelized
            self.t_env.sample_task_state()
            mlf = self.a_space.sample_ml_framework(mlf_signature)
            mlf, reward, score = self.t_env.evaluate(mlf)
            if reward is None:
                reward = self.t_env.error_reward
            else:
                mlfs.append(mlf)
                scores.append(score)
            rewards.append(reward)
            if verbose:
                print(
                    "iter %d - n valid mlf: %d/%d" % (
                        i, len(mlfs), i + 1),
                    sep=" ", end="\r", flush=True)
        return mlfs, rewards, scores

    def end_episode(self, i_episode, n_iter, mlfs, rewards, scores):
        # accumulate stats
        n_successful_mlfs = len(mlfs)
        n_unique_mlfs = len(set(
            [tuple(mlf.named_steps.keys()) for mlf in mlfs]))
        n_unique_hyperparams = len(set(
            [str(mlf.get_params().items()) for mlf in mlfs]))
        mlf_diversity = utils._diversity_metric(
            n_unique_mlfs, n_successful_mlfs)
        hyperparam_diversity = utils._diversity_metric(
            n_unique_hyperparams, n_successful_mlfs)

        # NOTE that the random search fitter doesn't use rewards but it's
        # collect in order to compare with CASH agent.
        mean_rewards = np.mean(rewards)
        if len(scores) > 0:
            mean_val_scores = np.mean(scores)
            std_val_scores = np.std(scores)
            best_validation_score = np.max(scores)
            best_mlf = mlfs[np.argmax(scores)]
        else:
            mean_val_scores, std_val_scores = np.nan, np.nan
            best_validation_score, best_mlf = np.nan, np.nan

        self.tracker.update_metrics({
            "episode": i_episode + 1,
            "data_env_names": self.t_env.current_data_env.name,
            "scorers": self.t_env.scorer.name,
            "losses": np.nan,
            "mean_rewards": mean_rewards,
            "aggregate_gradients": np.nan,
            "mean_validation_scores": mean_val_scores,
            "std_validation_scores": std_val_scores,
            "best_validation_scores": best_validation_score,
            "best_mlfs": best_mlf,
            "n_successful_mlfs": n_successful_mlfs,
            "n_unique_mlfs": n_unique_mlfs,
            "n_unique_hyperparams": n_unique_hyperparams,
            "mlf_diversity": mlf_diversity,
            "hyperparam_diversity": hyperparam_diversity
        })
        if self._metrics_logger is not None:
            self._metrics_logger(self.tracker)

    @property
    def history(self):
        return self.tracker.history

    @property
    def best_mlfs(self):
        return self.tracker.best_mlfs
