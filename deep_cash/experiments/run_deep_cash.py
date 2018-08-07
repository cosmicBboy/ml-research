"""Run deep cash experiment."""

import click
import os
import pandas as pd
import torch
import torch.multiprocessing as mp

from functools import partial
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
@click.option("--n_trials", default=1)
@click.option("--input_size", default=30)
@click.option("--hidden_size", default=30)
@click.option("--output_size", default=30)
@click.option("--n_layers", default=3)
@click.option("--dropout_rate", default=0.2)
@click.option("--beta", default=0.9)
@click.option("--with_baseline/--without_baseline", default=True)
@click.option("--single_baseline/--multi_baseline", default=True)
@click.option("--normalize_reward", default=False, is_flag=True)
@click.option("--n_episodes", default=1000)
@click.option("--n_iter", default=10)
@click.option("--learning_rate", default=0.005, type=float)
@click.option("--error_reward", default=0, type=float)
@click.option("--per_framework_time_limit", default=60)
@click.option("--per_framework_memory_limit", default=3077)
@click.option("--logger", default=None)
@click.option("--fit_verbose", default=1)
@click.option("--controller_seed", default=1000)
@click.option("--task_environment_seed", default=100)
def run_experiment(
        datasets, output_fp, n_trials, input_size, hidden_size, output_size,
        n_layers, dropout_rate, beta, with_baseline, single_baseline,
        normalize_reward, n_episodes, n_iter, learning_rate, error_reward,
        per_framework_time_limit, per_framework_memory_limit, logger,
        fit_verbose, controller_seed, task_environment_seed):
    """Run deep cash experiment with single configuration."""
    print("Running cash controller experiment with %d %s" % (
        n_trials, "trials" if n_trials > 1 else "trial"))
    torch.manual_seed(controller_seed)
    datasets = list(datasets)
    for ds in datasets:
        if ds not in ENV_NAMES:
            raise click.UsageError(
                "dataset %s is not a valid dataset name, options: %s" % (
                    ds, ENV_NAMES))
    output_fp = os.path.dirname(__file__) + "/../output" if \
        output_fp is None else output_fp
    data_path = Path(output_fp)
    logger = get_loggers().get(logger, empty_logger)
    metafeatures_dim = get_metafeatures_dim()

    def worker(procnum, reinforce, return_dict):
        """Fit REINFORCE Helper function."""
        reinforce.fit(
            optim=torch.optim.Adam,
            optim_kwargs={"lr": learning_rate},
            n_episodes=n_episodes,
            n_iter=n_iter,
            verbose=bool(int(fit_verbose)),
            procnum=procnum)
        # serialize reinforce controller here
        reinforce.controller.save(data_path / ("controller_trial_%d.pt" % i))
        return_dict[procnum] = reinforce.history()

    # multiprocessing manager
    manager = mp.Manager()
    return_dict = manager.dict()
    processes = []

    for i in range(n_trials):
        t_env = TaskEnvironment(
            f1_score,
            scorer_kwargs={"average": "weighted"},
            random_state=task_environment_seed,
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
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            a_space=a_space,
            dropout_rate=dropout_rate,
            num_rnn_layers=n_layers)

        reinforce = CASHReinforce(
            controller, t_env,
            beta=beta,
            with_baseline=with_baseline,
            single_baseline=single_baseline,
            normalize_reward=normalize_reward,
            metrics_logger=partial(logger, prefix="proc_num_%d_" % i))

        p = mp.Process(target=worker, args=(i, reinforce, return_dict))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    history = []
    for i, h in return_dict.items():
        h = pd.DataFrame(h).assign(trial_number=i)
        history.append(h)
        # save best mlfs per trial per episode
        mlf_path = data_path / ("cash_controller_mlfs_trial_%d" % i)
        if mlf_path.exists():
            rmtree(mlf_path)
        mlf_path.mkdir()
        for ep, mlf in enumerate(h.best_mlfs):
            joblib.dump(mlf, mlf_path / ("best_mlf_episode_%d.pkl" % (ep + 1)))
    history = pd.concat(history)

    # save history of all trials
    history[[
        "episode",
        "data_env_names",
        "losses",
        "mean_rewards",
        "mean_validation_scores",
        "std_validation_scores",
        "n_successful_mlfs",
        "n_unique_mlfs",
        "n_unique_hyperparameters",
        "best_validation_scores",
        "trial_number",
    ]].to_csv(
        str(data_path / "rnn_cash_controller_experiment.csv"),
        index=False)


if __name__ == "__main__":
    run_experiment()
