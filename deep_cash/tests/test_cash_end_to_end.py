"""End to end test for fitting a cash controllers"""

import pandas as pd
import torch

from sklearn.metrics import f1_score

from deep_cash.algorithm_space import AlgorithmSpace
from deep_cash.task_environment import TaskEnvironment
from deep_cash.cash_controller import CASHController
from deep_cash.cash_reinforce import CASHReinforce
from deep_cash.utils import get_metafeatures_dim


def test_cash_reinforce_fit():
    """Ensure DeepCASH training routine executes."""

    n_episodes = 3

    # hyperparameters
    t_env = TaskEnvironment(
        f1_score,
        scorer_kwargs={"average": "weighted"},
        random_state=100,
        per_framework_time_limit=10,
        per_framework_memory_limit=1000,
        dataset_names=["iris"],
        error_reward=0,
        reward_transformer=lambda x: x)

    # create algorithm space
    a_space = AlgorithmSpace(
        with_end_token=False,
        hyperparam_with_start_token=False,
        hyperparam_with_none_token=False)

    controller = CASHController(
        metafeature_size=get_metafeatures_dim(),
        # TODO: don't need input_size anymore, since a_space is an argument.
        input_size=a_space.n_components,
        hidden_size=10,
        output_size=10,
        a_space=a_space,
        dropout_rate=0.2,
        num_rnn_layers=3)

    reinforce = CASHReinforce(
        controller,
        t_env,
        with_baseline=False,
        metrics_logger=None)

    reinforce.fit(
        optim=torch.optim.Adam,
        optim_kwargs={"lr": 0.005},
        n_episodes=n_episodes,
        n_iter=3,
        verbose=True)

    history = pd.DataFrame(reinforce.history())
    assert history.shape[0] == n_episodes
    # assert best_validation_scores, mean_rewards, mean_validation_scores are
    # in [0, 1] range
