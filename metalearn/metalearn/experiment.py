"""Module for creating, reading deep cash experiments."""

import datetime
import inspect
import itertools
import joblib
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

from .algorithm_space import AlgorithmSpace
from .evaluate import evaluate_controller
from .task_environment import TaskEnvironment
from .metalearn_controller import MetaLearnController
from .metalearn_reinforce import MetaLearnReinforce
from .data_environments.environments import envs
from .data_types import ExperimentType
from .loggers import get_loggers, empty_logger
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
        "parameters",
        "notes",
    ])


def _env_names():
    return [d["dataset_name"] for d in envs()]


def get_experiment_fn(experiment_type):
    return {
        ExperimentType.METALEARN_REINFORCE: run_experiment,
    }[ExperimentType[experiment_type]]


def get_default_parameters(experiment_type):
    experiment_fn = get_experiment_fn(experiment_type)
    return OrderedDict([
        (k, v.default) for k, v in
        inspect.signature(experiment_fn).parameters.items()
    ])


def create_config(name, experiment_type, description=None,
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
        parameters=parameters,
        notes=None,
    )


def write_config(config, dir_path, fname=None):
    dir_path = Path(dir_path)
    dir_path.mkdir(exist_ok=True)
    fp = _config_filepath(dir_path, config) if fname is None \
        else dir_path / fname
    with fp.open("w") as f:
        yaml.dump(config._asdict(), f, Dumper=yamlordereddictloader.Dumper,
                  default_flow_style=False)
    print("wrote experiment config file to %s" % fp)


def read_config(fp):
    with open(fp, "r") as f:
        config_dict = yaml.load(f, Loader=yamlordereddictloader.Loader)
    if "notes" not in config_dict:
        config_dict["notes"] = None
    return ExperimentConfig(**config_dict)


def _config_filepath(dir_path, config):
    return dir_path / (
        "experiment_%s_%s.yml" % (config.created_at, config.name))


def save_best_mlfs(data_path, best_mlfs, proc_num):
    mlf_path = data_path / ("metalearn_controller_mlfs_trial_%d" % proc_num)
    mlf_path.mkdir(exist_ok=True)
    for i, mlf in enumerate(best_mlfs):
        joblib.dump(
            mlf, mlf_path / ("best_mlf_episode_%d.pkl.gz" % (i + 1)),
            compress=3,
        )


def create_hyperparameter_grid(hyperparameters):
    allowable_hyperparameters = [
        "input_size",
        "hidden_size",
        "output_size",
        "n_layers",
        "n_episodes",
        "entropy_coef",
        "entropy_coef_anneal_to",
        "learning_rate",
        "meta_reward_multiplier",
        "normalize_reward",
        "error_reward",
    ]

    if isinstance(hyperparameters, dict):
        hyperparameters = {
            k: v for k, v in hyperparameters.items() if v is not None
        }
        hyperparameter_grid = itertools.product(*[
            [(hyperparameter, value)
             for value in hyperparameters[hyperparameter]]
            for hyperparameter in allowable_hyperparameters
            if hyperparameter in hyperparameters
        ])
        return list(map(dict, hyperparameter_grid))
    elif isinstance(hyperparameters, list):
        return [
            {k: v for k, v in h_dict.items() if k in allowable_hyperparameters}
            for h_dict in hyperparameters
        ]
    else:
        raise ValueError(
            "hyperparameters type %s not recognized" % type(hyperparameters)
        )


def run_experiment(
        datasets=None,
        test_datasets=None,
        output_fp=os.path.dirname(__file__) + "/../output",
        input_size=30,
        hidden_size=30,
        output_size=30,
        n_layers=3,
        dropout_rate=0.2,
        entropy_coef=0.0,
        entropy_coef_anneal_to=0.0,
        entropy_coef_anneal_by=None,
        normalize_reward=True,
        gamma=0.99,
        meta_reward_multiplier=1.,
        n_episodes=100,
        n_iter=16,
        n_eval_iter=10,
        n_eval_samples=5,
        learning_rate=0.005,
        optim_beta1=0.9,
        optim_beta2=0.999,
        env_sources=["SKLEARN", "OPEN_ML", "KAGGLE"],
        test_env_sources=["AUTOSKLEARN_BENCHMARK"],
        target_types=["BINARY", "MULTICLASS"],
        test_env_target_types=["BINARY", "MULTICLASS"],
        test_set_config={
            "SKLEARN": {
                "test_size": 0.8,
                "random_state": 100,
            },
            "AUTOSKLEARN_BENCHMARK": {
                "test_size": 0.8,
                "random_state": 100,
            },
        },
        error_reward=0,
        n_samples=5000,
        per_framework_time_limit=180,
        per_framework_memory_limit=5000,
        metric_logger=None,
        fit_verbose=1,
        controller_seed=1000,
        task_environment_seed=100,
        hyperparameters=None):
    """Run deep cash experiment with single configuration."""
    torch.manual_seed(controller_seed)
    output_fp = os.path.dirname(__file__) + "/../output" if \
        output_fp is None else output_fp
    data_path = Path(output_fp)
    data_path.mkdir(exist_ok=True)
    hyperparameters = {} if hyperparameters is None else hyperparameters

    # initialize error logging (this is to log fit/predict/score errors made
    # when evaluating a proposed MLF)
    utils.init_logging(str(Path(output_fp) / "fit_predict_error_logs.log"))

    # this logger is for logging metrics to floydhub/stdout
    metric_logger = get_loggers().get(metric_logger, empty_logger)

    def fit_metalearn(proc_num, reinforce, **hyperparameters):
        """Fit Metalearning Algorithm helper."""
        reinforce.t_env.reset_random_state()
        reinforce.fit(
            optim=torch.optim.Adam,
            optim_kwargs={
                "lr": hyperparameters.get("learning_rate", learning_rate),
                "betas": (optim_beta1, optim_beta2),
            },
            n_episodes=hyperparameters.get("n_episodes", n_episodes),
            n_iter=n_iter,
            verbose=bool(int(fit_verbose)),
            procnum=proc_num)

        # serialize reinforce controller
        reinforce.controller.save(
            data_path / f"controller_trial_{proc_num}.pt")

        # serialize best mlfs
        save_best_mlfs(data_path, reinforce.best_mlfs, proc_num)

        evaluation_results = evaluate_controller(
            reinforce.controller,
            reinforce.t_env,
            meta_reward_multiplier=hyperparameters.get(
                "meta_reward_multiplier", meta_reward_multiplier),
            n_shots=n_eval_iter,
            n_eval_samples=n_eval_samples,
        )

        for env_key, results in evaluation_results.items():
            if results is None:
                continue
            fname = f"{env_key}_inference_results_trial_{proc_num}.csv"
            with (data_path / fname).open("w") as f:
                results.to_csv(f, index=False)

        history = pd.DataFrame(reinforce.history).assign(
            trial_num=proc_num, **hyperparameters)
        history.to_csv(
            data_path / f"metalearn_training_results_trial_{proc_num}.csv",
            index=False)

    processes = []
    hyperparameter_grid = create_hyperparameter_grid(hyperparameters)
    print("hyperparameter grid: %s" % hyperparameter_grid)
    for proc_num, _hyperparameters in enumerate(hyperparameter_grid):
        t_env = TaskEnvironment(
            env_sources=env_sources,
            test_env_sources=test_env_sources,
            target_types=target_types,
            test_env_target_types=test_env_target_types,
            test_set_config=test_set_config,
            random_state=task_environment_seed,
            per_framework_time_limit=per_framework_time_limit,
            per_framework_memory_limit=per_framework_memory_limit,
            dataset_names=datasets,
            test_dataset_names=test_datasets,
            error_reward=_hyperparameters.get("error_reward", error_reward),
            n_samples=n_samples)

        a_space = AlgorithmSpace(
            with_end_token=False,
            hyperparam_with_start_token=False,
            hyperparam_with_none_token=False)

        controller = MetaLearnController(
            metafeature_size=t_env.metafeature_dim,
            input_size=_hyperparameters.get("input_size", input_size),
            hidden_size=_hyperparameters.get("hidden_size", hidden_size),
            output_size=_hyperparameters.get("output_size", output_size),
            a_space=a_space,
            dropout_rate=dropout_rate,
            num_rnn_layers=_hyperparameters.get("n_layers", n_layers),
        )

        reinforce = MetaLearnReinforce(
            controller,
            t_env,
            gamma=gamma,
            entropy_coef=_hyperparameters.get("entropy_coef", entropy_coef),
            entropy_coef_anneal_to=_hyperparameters.get(
                "entropy_coef_anneal_to", entropy_coef_anneal_to),
            entropy_coef_anneal_by=entropy_coef_anneal_by,
            normalize_reward=_hyperparameters.get(
                "normalize_reward", normalize_reward),
            meta_reward_multiplier=_hyperparameters.get(
                "meta_reward_multiplier", meta_reward_multiplier),
            metrics_logger=partial(
                metric_logger, prefix="proc_num_%d" % proc_num)
        )

        p = mp.Process(
            target=fit_metalearn,
            args=(proc_num, reinforce),
            kwargs=_hyperparameters,
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    for i in range(len(hyperparameter_grid)):
        print("loading controllers from each trial")
        try:
            controller = reinforce.controller.load(
                data_path / ("controller_trial_%d.pt" % i))
            print(controller)
        except Exception:
            print("could not read controller for process %d" % i)
