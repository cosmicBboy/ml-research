"""Module for creating, reading deep cash experiments."""

import datetime
import inspect
import logging
import os
import pandas as pd
import subprocess
import torch
import torch.multiprocessing as mp
import yaml
import yamlordereddictloader

from collections import namedtuple, OrderedDict
from functools import partial
from pathlib import Path
from shutil import rmtree
from sklearn.externals import joblib

from .algorithm_space import AlgorithmSpace
from .task_environment import TaskEnvironment
from .cash_controller import CASHController
from .cash_reinforce import CASHReinforce
from .loggers import get_loggers, empty_logger
from .data_environments.environments import envs
from . import utils


logger = logging.getLogger(__name__)


ExperimentConfig = namedtuple(
    "ExperimentConfig", [
        "name",
        "description",
        "created_at",
        "git_hash",
        "parameters"])


def _env_names():
    return [d["dataset_name"] for d in envs()]


def get_default_parameters():
    return OrderedDict([
        (k, v.default) for k, v in
        inspect.signature(run_experiment).parameters.items()
    ])


def create_config(name, description, date_format="%Y-%m-%d-%H:%M:%S",
                  **custom_parameters):
    parameters = get_default_parameters()
    if custom_parameters:
        parameters.update(custom_parameters)
    created_at = datetime.datetime.now()
    git_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
    return ExperimentConfig(
        name=name,
        description=description,
        created_at=created_at.strftime(date_format),
        git_hash=git_hash.strip().decode("utf-8"),
        parameters=parameters)


def write_config(config, dir_path):
    dir_path = Path(dir_path)
    dir_path.mkdir(exist_ok=True)
    fp = _config_filepath(dir_path, config)
    with open(str(fp), "w") as f:
        yaml.dump(config._asdict(), f, Dumper=yamlordereddictloader.Dumper,
                  default_flow_style=False)
    print("wrote experiment config file to %s" % fp)


def read_config(fp):
    with open(fp, "r") as f:
        return yaml.load(f, Loader=yamlordereddictloader.Loader)


def _config_filepath(dir_path, config):
    return dir_path / (
        "experiment_%s_%s.yml" % (config.created_at, config.name))


def run_experiment(
        datasets=None,
        output_fp=os.path.dirname(__file__) + "/../output",
        n_trials=1,
        input_size=30,
        hidden_size=30,
        output_size=30,
        n_layers=3,
        dropout_rate=0.2,
        beta=0.9,
        entropy_coef=0.0,
        with_baseline=True,
        single_baseline=True,
        normalize_reward=False,
        n_episodes=100,
        n_iter=16,
        learning_rate=0.005,
        target_types=["BINARY", "MULTICLASS"],
        error_reward=0,
        per_framework_time_limit=180,
        per_framework_memory_limit=5000,
        metric_logger=None,
        fit_verbose=1,
        controller_seed=1000,
        task_environment_seed=100):
    """Run deep cash experiment with single configuration."""
    print("Running cash controller experiment with %d %s" % (
        n_trials, "trials" if n_trials > 1 else "trial"))
    torch.manual_seed(controller_seed)
    env_names = _env_names()
    if datasets:
        datasets = list(datasets)
        for ds in datasets:
            if ds not in env_names:
                raise ValueError(
                    "dataset %s is not a valid dataset name, options: %s" % (
                        ds, env_names))
    output_fp = os.path.dirname(__file__) + "/../output" if \
        output_fp is None else output_fp
    data_path = Path(output_fp)

    # initialize error logging (this is to log fit/predict/score errors made
    # when evaluating a proposed MLF)
    utils.init_logging(str(Path(output_fp) / "fit_predict_error_logs.log"))

    # this logger is for logging metrics to floydhub/stdout
    metric_logger = get_loggers().get(metric_logger, empty_logger)

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
            target_types=target_types,
            random_state=task_environment_seed,
            per_framework_time_limit=per_framework_time_limit,
            per_framework_memory_limit=per_framework_memory_limit,
            dataset_names=datasets,
            error_reward=error_reward)

        # create algorithm space
        a_space = AlgorithmSpace(
            with_end_token=False,
            hyperparam_with_start_token=False,
            hyperparam_with_none_token=False)

        controller = CASHController(
            metafeature_size=t_env.metafeature_dim,
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            a_space=a_space,
            dropout_rate=dropout_rate,
            num_rnn_layers=n_layers)

        reinforce = CASHReinforce(
            controller, t_env,
            beta=beta,
            entropy_coef=entropy_coef,
            with_baseline=with_baseline,
            single_baseline=single_baseline,
            normalize_reward=normalize_reward,
            metrics_logger=partial(metric_logger, prefix="proc_num_%d_" % i))

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
        "aggregate_gradients",
        "mean_rewards",
        "mean_validation_scores",
        "std_validation_scores",
        "n_successful_mlfs",
        "n_unique_mlfs",
        "n_unique_hyperparams",
        "mlf_diversity",
        "hyperparam_diversity",
        "best_validation_scores",
        "trial_number",
    ]].to_csv(
        str(data_path / "rnn_cash_controller_experiment.csv"),
        index=False)
