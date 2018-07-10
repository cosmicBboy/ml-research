"""Run deep cash experiment."""

import click
import os
import pandas as pd
import torch

from pathlib import Path
from shutil import rmtree
from sklearn.externals import joblib
from sklearn.metrics import f1_score

from deep_cash.algorithm_space import AlgorithmSpace
from deep_cash.task_environment import TaskEnvironment
from deep_cash.cash_controller import CASHController
from deep_cash.cash_reinforce import CASHReinforce
from deep_cash.loggers import get_loggers, empty_logger
from deep_cash.utils import get_metafeatures_dim
from deep_cash.data_environments.classification_environments import env_names

DEFAULT_OUTPUT = os.path.dirname(__file__) + "/../output"
ENV_NAMES = env_names()


@click.command()
@click.argument("datasets", nargs=-1)
@click.option("--output_fp", default=DEFAULT_OUTPUT)
@click.option("--hidden_size", default=30)
@click.option("--output_size", default=30)
@click.option("--n_layers", default=3)
@click.option("--dropout_rate", default=0.2)
@click.option("--beta", default=0.9)
@click.option("--n_episodes", default=1000)
@click.option("--n_iter", default=10)
@click.option("--learning_rate", default=0.005)
@click.option("--error_reward", default=0)
@click.option("--per_framework_time_limit", default=60)
@click.option("--per_framework_memory_limit", default=3077)
@click.option("--logger", default=None)
@click.option("--fit_verbose", default=1)
def run_experiment(
        datasets, output_fp, hidden_size, output_size, n_layers, dropout_rate,
        beta, n_episodes, n_iter, learning_rate, error_reward,
        per_framework_time_limit, per_framework_memory_limit, logger,
        fit_verbose):
    """Run deep cash experiment with single configuration."""
    datasets = list(datasets)
    for ds in datasets:
        if ds not in ENV_NAMES:
            raise click.UsageError(
                "dataset %s is not a valid dataset name, options: %s" % (
                    ds, ENV_NAMES))
    output_fp = os.path.dirname(__file__) + "/../output" if \
        output_fp is None else output_fp
    data_path = Path(output_fp)
    exp_path = data_path / "anneal_dataset"
    exp_path.mkdir(exist_ok=True)
    logger = get_loggers().get(logger, empty_logger)
    metafeatures_dim = get_metafeatures_dim()

    t_env = TaskEnvironment(
        f1_score,
        scorer_kwargs={"average": "weighted"},
        random_state=100,
        per_framework_time_limit=per_framework_time_limit,
        per_framework_memory_limit=per_framework_memory_limit,
        dataset_names=datasets,
        error_reward=error_reward,
        reward_transformer=lambda x: x)

    # create algorithm space
    a_space = AlgorithmSpace(
        with_end_token=False,
        hyperparam_with_start_token=False,
        hyperparam_with_none_token=False)

    controller = CASHController(
        metafeature_size=metafeatures_dim,
        input_size=a_space.n_components,
        hidden_size=hidden_size,
        output_size=output_size,
        a_space=a_space,
        optim=torch.optim.Adam,
        optim_kwargs={"lr": learning_rate},
        dropout_rate=dropout_rate,
        num_rnn_layers=n_layers)

    reinforce = CASHReinforce(
        controller, t_env, beta=beta, with_baseline=False,
        metrics_logger=logger)
    reinforce.fit(n_episodes=n_episodes, n_iter=n_iter, verbose=fit_verbose)

    history = pd.DataFrame(reinforce.history())
    history[[
        "episode",
        "data_env_names",
        "losses",
        "mean_rewards",
        "mean_validation_scores",
        "std_validation_scores",
        "n_successful_mlfs",
        "n_unique_mlfs",
        "best_validation_scores",
    ]].to_csv(
        str(data_path / "rnn_cash_controller_experiment.csv"), index=False)

    mlf_path = data_path / "rnn_cash_controller_experiment_mlfs"
    if mlf_path.exists():
        rmtree(mlf_path)
    mlf_path.mkdir()
    for i, mlf in enumerate(history.best_mlfs):
        joblib.dump(mlf, mlf_path / ("best_mlf_episode_%d.pkl" % (i + 1)))

    controller.save(data_path / "rnn_cash_controller_experiment.pt")


if __name__ == "__main__":
    run_experiment()
