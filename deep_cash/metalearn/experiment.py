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
from .metalearn_controller import MetaLearnController
from .metalearn_reinforce import MetaLearnReinforce
from .data_environments.environments import envs
from .data_types import ExperimentType
from .loggers import get_loggers, empty_logger
from .random_search import CASHRandomSearch
from . import utils

# TODO: support ability to copy a config file with new name
# TODO: support ability to run floyd job by experiment name


logger = logging.getLogger(__name__)


ExperimentConfig = namedtuple(
    "ExperimentConfig", [
        "name",
        "experiment_type",
        "description",
        "created_at",
        "git_hash",
        "parameters"])


def _env_names():
    return [d["dataset_name"] for d in envs()]


def get_experiment_fn(experiment_type):
    return {
        ExperimentType.METALEARN_REINFORCE: run_experiment,
        ExperimentType.RANDOM_SEARCH: run_random_search,
    }[ExperimentType[experiment_type]]


def get_default_parameters(experiment_type):
    experiment_fn = get_experiment_fn(experiment_type)
    return OrderedDict([
        (k, v.default) for k, v in
        inspect.signature(experiment_fn).parameters.items()
    ])


def create_config(name, experiment_type, description,
                  date_format="%Y-%m-%d-%H-%M-%S", **custom_parameters):
    parameters = get_default_parameters(experiment_type)
    if custom_parameters:
        parameters.update(custom_parameters)
    created_at = datetime.datetime.now()
    git_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
    return ExperimentConfig(
        name=name,
        experiment_type=experiment_type,
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


def gather_history(return_dict):
    history = []
    for i, h in return_dict.items():
        h = pd.DataFrame(h).assign(trial_number=i)
        history.append(h)
    return pd.concat(history)


def save_best_mlfs(data_path, best_mlfs, procnum):
    mlf_path = data_path / ("metalearn_controller_mlfs_trial_%d" % procnum)
    mlf_path.mkdir(exist_ok=True)
    for i, mlf in enumerate(best_mlfs):
        joblib.dump(mlf, mlf_path / ("best_mlf_episode_%d.pkl" % (i + 1)))


def save_experiment(history, data_path, fname):
    # save history of all trials
    history.to_csv(str(data_path / fname), index=False)


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
        env_sources=["SKLEARN", "OPEN_ML", "KAGGLE"],
        target_types=["BINARY", "MULTICLASS"],
        test_set_config=OrderedDict([
            ("SKLEARN", OrderedDict([
                ("test_size", 0.8),
                ("random_state", 100)]))
        ]),
        error_reward=0,
        n_samples=5000,
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
        # serialize best mlfs
        save_best_mlfs(data_path, reinforce.best_mlfs, procnum)
        return_dict[procnum] = reinforce.history

    # multiprocessing manager
    manager = mp.Manager()
    return_dict = manager.dict()
    processes = []

    for i in range(n_trials):
        t_env = TaskEnvironment(
            env_sources=env_sources,
            target_types=target_types,
            test_set_config=test_set_config,
            random_state=task_environment_seed,
            per_framework_time_limit=per_framework_time_limit,
            per_framework_memory_limit=per_framework_memory_limit,
            dataset_names=datasets,
            error_reward=error_reward,
            n_samples=n_samples)

        # create algorithm space
        a_space = AlgorithmSpace(
            with_end_token=False,
            hyperparam_with_start_token=False,
            hyperparam_with_none_token=False)

        controller = MetaLearnController(
            metafeature_size=t_env.metafeature_dim,
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            a_space=a_space,
            dropout_rate=dropout_rate,
            num_rnn_layers=n_layers)

        reinforce = MetaLearnReinforce(
            controller, t_env,
            beta=beta,
            entropy_coef=entropy_coef,
            with_baseline=with_baseline,
            single_baseline=single_baseline,
            normalize_reward=normalize_reward,
            metrics_logger=partial(metric_logger, prefix="proc_num_%d" % i))

        p = mp.Process(target=worker, args=(i, reinforce, return_dict))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    save_experiment(
        gather_history(return_dict),
        data_path, "rnn_metalearn_controller_experiment.csv")


def run_random_search(
        datasets=None,
        output_fp=os.path.dirname(__file__) + "/../output",
        n_trials=1,
        n_episodes=100,
        n_iter=16,
        env_sources=["SKLEARN", "OPEN_ML", "KAGGLE"],
        target_types=["BINARY", "MULTICLASS"],
        test_set_config=OrderedDict([
            ("SKLEARN", OrderedDict([
                ("test_size", 0.8),
                ("random_state", 100)]))
        ]),
        error_reward=0,
        n_samples=5000,
        per_framework_time_limit=180,
        per_framework_memory_limit=5000,
        metric_logger="default",
        fit_verbose=1,
        controller_seed=1000,
        task_environment_seed=100):
    """Run deep cash experiment with single configuration."""
    print("Running random search experiment with %d %s" % (
        n_trials, "trials" if n_trials > 1 else "trial"))
    output_fp = os.path.dirname(__file__) + "/../output" if \
        output_fp is None else output_fp
    data_path = Path(output_fp)

    # initialize error logging (this is to log fit/predict/score errors made
    # when evaluating a proposed MLF)
    utils.init_logging(
        str(Path(output_fp) / "fit_predict_error_logs_randomsearch.log"))

    # this logger is for logging metrics to floydhub/stdout
    metric_logger = get_loggers().get(metric_logger, empty_logger)

    def worker(procnum, random_search, return_dict):
        """Fit REINFORCE Helper function."""
        random_search.fit(
            n_episodes=n_episodes,
            n_iter=n_iter,
            verbose=fit_verbose,
            procnum=procnum)
        save_best_mlfs(data_path, random_search.best_mlfs, procnum)
        return_dict[procnum] = random_search.history

    # multiprocessing manager
    manager = mp.Manager()
    return_dict = manager.dict()
    processes = []

    for i in range(n_trials):
        t_env = TaskEnvironment(
            env_sources=env_sources,
            target_types=target_types,
            test_set_config=test_set_config,
            random_state=task_environment_seed,
            per_framework_time_limit=per_framework_time_limit,
            per_framework_memory_limit=per_framework_memory_limit,
            dataset_names=datasets,
            error_reward=error_reward,
            n_samples=n_samples)

        # create algorithm space
        a_space = AlgorithmSpace(
            with_end_token=False,
            hyperparam_with_start_token=False,
            hyperparam_with_none_token=False)

        random_search = CASHRandomSearch(
            a_space=a_space,
            t_env=t_env,
            metrics_logger=partial(metric_logger, prefix="proc_num_%d" % i))

        p = mp.Process(target=worker, args=(i, random_search, return_dict))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    save_experiment(
        gather_history(return_dict),
        data_path, "rnn_cash_randomsearch_experiment.csv")
