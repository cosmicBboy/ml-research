"""End to end tests for fitting a cash controllers."""

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import f1_score

from deep_cash.algorithm_space import AlgorithmSpace
from deep_cash.task_environment import TaskEnvironment
from deep_cash.cash_controller import CASHController
from deep_cash.cash_reinforce import CASHReinforce


def _task_environment():
    return TaskEnvironment(
        env_sources=["sklearn"],
        scorer=f1_score,
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


def _cash_controller(a_space, t_env):
    return CASHController(
        metafeature_size=t_env.metafeature_dim,
        input_size=20,
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
    controller = _cash_controller(a_space, t_env)
    reinforce = _cash_reinforce(controller, t_env, with_baseline=True)
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
    controller = _cash_controller(a_space, t_env)
    reinforce = _cash_reinforce(
        controller, t_env, with_baseline=True, single_baseline=False)
    reinforce.fit(
        n_episodes=n_episodes,
        **_fit_kwargs())
    assert reinforce._baseline_buffer_history["iris"] != \
        reinforce._baseline_buffer_history["wine"]


def test_cash_zero_gradient():
    """Test that gradient is zero if the reward is zero."""
    torch.manual_seed(100)  # ensure weight initialized is deterministic
    n_episodes = 20
    t_env = _task_environment()
    a_space = _algorithm_space()
    controller = _cash_controller(a_space, t_env)
    # don't train with baseline since this will modify the reward signal when
    # computing the `advantage = reward - baseline`.
    reinforce = _cash_reinforce(controller, t_env, with_baseline=False)
    fit_kwargs = _fit_kwargs()
    fit_kwargs.update({"n_iter": 1})
    reinforce.fit(
        n_episodes=n_episodes,
        **_fit_kwargs())
    # make sure there's at least one zero-valued aggregate gradient
    assert any([g == 0 for g in reinforce.aggregate_gradients])
    for r, g in zip(reinforce.mean_rewards, reinforce.aggregate_gradients):
        if r == 0:
            assert g == 0
        else:
            assert g != 0


def test_cash_entropy_regularizer():
    """Test that losses w/ entropy regularization are lower than baseline."""
    losses = {}
    for model, kwargs in [
            ("baseline", {
                "with_baseline": True,
                "entropy_coef": 0.0}),
            ("entropy_regularized", {
                "with_baseline": True,
                "entropy_coef": 0.5})]:
        torch.manual_seed(100)  # ensure weight initialized is deterministic
        # only run for a few episodes because model losses
        # become incomparable as models diverge
        n_episodes = 3
        t_env = _task_environment()
        a_space = _algorithm_space()
        controller = _cash_controller(a_space, t_env)
        reinforce = _cash_reinforce(controller, t_env, **kwargs)
        fit_kwargs = _fit_kwargs()
        fit_kwargs.update({"n_iter": 4})
        reinforce.fit(
            n_episodes=n_episodes,
            **_fit_kwargs())
        losses[model] = reinforce.losses
    assert (
        np.array(losses["entropy_regularized"]) <
        np.array(losses["baseline"])).all()
