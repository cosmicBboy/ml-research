"""Example usage of the CASH Controller."""

import logging

import os
import pandas as pd
import torch

from shutil import rmtree
from sklearn.externals import joblib

from deep_cash.task_environment import TaskEnvironment
from deep_cash.algorithm_space import AlgorithmSpace
from deep_cash.cash_controller import CASHController
from deep_cash.cash_reinforce import CASHReinforce
from deep_cash.components import classifiers
from deep_cash import utils

utils.init_logging(None)

logger = logging.getLogger(__name__)
data_path = os.path.dirname(__file__) + "/artifacts"

# hyperparameters
n_episodes = 100
n_iter = 10
learning_rate = 0.003
error_reward = 0
metrics_logger = None
fit_verbose = True

hidden_size = 30
output_size = 30
n_layers = 3

torch.manual_seed(1000)


t_env = TaskEnvironment(
    env_sources=["SKLEARN"],
    dataset_names=["sklearn.digits"],
    random_state=100,
    enforce_limits=True,
    per_framework_time_limit=720,
    per_framework_memory_limit=10000,
    error_reward=error_reward,
    target_types=["BINARY", "MULTICLASS"])

# create algorithm space
a_space = AlgorithmSpace(
    classifiers=[classifiers.logistic_regression()],
    with_end_token=False,
    hyperparam_with_start_token=False,
    hyperparam_with_none_token=False)

controller = CASHController(
    metafeature_size=t_env.metafeature_dim,
    input_size=a_space.n_components,
    hidden_size=hidden_size,
    output_size=output_size,
    a_space=a_space,
    dropout_rate=0.2,
    num_rnn_layers=n_layers)

reinforce = CASHReinforce(
    controller,
    t_env,
    with_baseline=False,
    metrics_logger=metrics_logger)

reinforce.fit(
    optim=torch.optim.Adam,
    optim_kwargs={"lr": learning_rate},
    n_episodes=n_episodes,
    n_iter=n_iter,
    verbose=fit_verbose)


history = pd.DataFrame(reinforce.history)
history.to_csv(
    str(data_path / "rnn_cash_controller_experiment.csv"), index=False)

mlf_path = data_path / "rnn_cash_controller_experiment_mlfs"
if mlf_path.exists():
    rmtree(mlf_path)
mlf_path.mkdir()
for i, mlf in enumerate(reinforce.best_mlfs):
    joblib.dump(mlf, mlf_path / ("best_mlf_episode_%d.pkl" % (i + 1)))

controller.save(data_path / "rnn_cash_controller_experiment.pt")
