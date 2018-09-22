"""Experiment with tuning exponential mean `beta` controller hyperparameter."""

import functools
import pandas as pd
import os
import torch
import torch.multiprocessing as mp

from pathlib import Path
from sklearn.metrics import f1_score

from deep_cash.algorithm_space import AlgorithmSpace
from deep_cash.task_environment import TaskEnvironment
from deep_cash.cash_controller import CASHController
from deep_cash.cash_reinforce import CASHReinforce
from deep_cash.loggers import get_loggers, empty_logger


data_path = Path(os.environ.get(
    "DEEP_CASH_OUT_PATH", os.path.dirname(__file__) + "/../output"))
exp_path = data_path / "tuning_exponential_mean_beta"
exp_path.mkdir(exist_ok=True)

# hyperparameters
n_episodes = int(os.environ.get("DEEP_CASH_N_EPISODES", 300))
n_iter = int(os.environ.get("DEEP_CASH_N_ITER", 100))
learning_rate = float(os.environ.get("DEEP_CASH_LEARNING_RATE", 0.005))
error_reward = int(os.environ.get("DEEP_CASH_ERROR_REWARD", -1))
logger_name = os.environ.get("DEEP_CASH_LOGGER", None)
logger = get_loggers().get(
    os.environ.get("DEEP_CASH_LOGGER", None), empty_logger)
fit_verbose = int(os.environ.get("DEEP_CASH_FIT_VERBOSE", 1))

hidden_size = 30
output_size = 30
n_layers = 3


t_env = TaskEnvironment(
    random_state=100,
    per_framework_time_limit=60,
    per_framework_memory_limit=3077,
    error_reward=error_reward)

# create algorithm space
a_space = AlgorithmSpace(
    with_end_token=False,
    hyperparam_with_start_token=False,
    hyperparam_with_none_token=False)


fit_kwargs = {
    "n_episodes": n_episodes,
    "n_iter": n_iter,
    "verbose": fit_verbose,
}


def worker(procnum, reinforce, return_dict):
    """Parallelize gridsearch helper function."""
    reinforce.fit(**fit_kwargs)
    reinforce.controller.save(exp_path / ("controller_%d.pt" % procnum))
    return_dict[procnum] = reinforce.history()


manager = mp.Manager()
return_dict = manager.dict()
num_processes = 4
processes = []
betas = [0.99, 0.9, 0.75, 0.5]
for i, beta in enumerate(betas):
    _logger = functools.partial(logger, prefix="beta_%0.02f" % beta)
    controller = CASHController(
        metafeature_size=t_env.metafeature_dim,
        input_size=a_space.n_components,
        hidden_size=hidden_size,
        output_size=output_size,
        a_space=a_space,
        optim=torch.optim.Adam,
        optim_kwargs={"lr": learning_rate},
        dropout_rate=0.2,
        num_rnn_layers=n_layers)
    reinforce = CASHReinforce(
        controller,
        t_env,
        beta=beta,
        with_baseline=False,
        metrics_logger=_logger)
    p = mp.Process(target=worker, args=(i, reinforce, return_dict))
    p.start()
    processes.append(p)
for p in processes:
    p.join()


histories = []
for i, h in return_dict.items():
    history = pd.DataFrame(h).assign(beta=betas[i])
    history[[
        "beta",
        "episode",
        "data_env_names",
        "losses",
        "mean_rewards",
        "mean_validation_scores",
        "std_validation_scores",
        "n_successful_mlfs",
        "n_unique_mlfs",
        "best_validation_scores",
    ]]
    histories.append(history)

histories = pd.concat(histories)
histories.to_csv(str(exp_path / "history.csv"), index=False)
