"""End to end tests for fitting a cash controllers."""

import pandas as pd
import torch

from sklearn.metrics import f1_score

from deep_cash.algorithm_space import AlgorithmSpace
from deep_cash.task_environment import TaskEnvironment
from deep_cash.cash_controller import CASHController
from deep_cash.cash_reinforce import CASHReinforce
from deep_cash.utils import get_metafeatures_dim


def _task_environment():
    return TaskEnvironment(
        f1_score,
        scorer_kwargs={"average": "weighted"},
        random_state=100,
        per_framework_time_limit=10,
        per_framework_memory_limit=1000,
        dataset_names=["iris", "wine"],
        error_reward=0,
        reward_transformer=lambda x: x)


def _algorithm_space():
    return AlgorithmSpace(
        with_end_token=False,
        hyperparam_with_start_token=False,
        hyperparam_with_none_token=False)


def _cash_controller(a_space):
    return CASHController(
        metafeature_size=get_metafeatures_dim(),
        # TODO: don't need input_size anymore, since a_space is an argument.
        input_size=a_space.n_components,
        hidden_size=10,
        output_size=10,
        a_space=a_space,
        dropout_rate=0.2,
        num_rnn_layers=3)


def _cash_reinforce(controller, task_env, **kwargs):
    return CASHReinforce(
        controller,
        task_env,
        beta=0.9,
        metrics_logger=None,
        with_baseline=True,
        **kwargs)


def _fit_kwargs():
    return {
        "optim": torch.optim.Adam,
        "optim_kwargs": {"lr": 0.005},
        "n_iter": 3,
        "verbose": False
    }


def test_cash_reinforce_fit():
    """Ensure DeepCASH training routine executes."""
    n_episodes = 4
    t_env = _task_environment()
    a_space = _algorithm_space()
    controller = _cash_controller(a_space)
    reinforce = _cash_reinforce(controller, t_env)
    reinforce.fit(
        n_episodes=n_episodes,
        **_fit_kwargs())

    history = pd.DataFrame(reinforce.history())
    assert history.shape[0] == n_episodes
    for metric in [
            "mean_rewards",
            "mean_validation_scores",
            "best_validation_scores"]:
        assert (history[metric].dropna() <= 1).all()
        assert (history[metric].dropna() >= 0).all()


def test_cash_reinforce_fit_multi_baseline():
    """Make sure that baseline function maintains buffers per data env."""
    n_episodes = 10
    t_env = _task_environment()
    a_space = _algorithm_space()
    controller = _cash_controller(a_space)
    reinforce = _cash_reinforce(controller, t_env, single_baseline=False)
    reinforce.fit(
        n_episodes=n_episodes,
        **_fit_kwargs())
    assert reinforce._baseline_buffer_history["iris"] != \
        reinforce._baseline_buffer_history["wine"]
