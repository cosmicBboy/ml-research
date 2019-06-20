"""Example usage of the CASH Controller."""

import logging

import os
import pandas as pd
import torch

from shutil import rmtree
from sklearn.externals import joblib

from metalearn.task_environment import TaskEnvironment
from metalearn.algorithm_space import AlgorithmSpace
from metalearn.random_search import CASHRandomSearch
from metalearn.components import classifiers
from metalearn import utils

utils.init_logging(None)

logger = logging.getLogger(__name__)
data_path = os.path.dirname(__file__) + "/artifacts"

# hyperparameters
n_episodes = 100
n_iter = 10
learning_rate = 0.003
error_reward = 0
logger = None
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

cash_random = CASHRandomSearch(a_space, t_env)
cash_random.fit(n_episodes=10, n_iter=20)


history = pd.DataFrame(cash_random.history)
history.to_csv(
    str(data_path / "rnn_metalearn_controller_experiment.csv"), index=False)

mlf_path = data_path / "rnn_metalearn_controller_experiment_mlfs"
if mlf_path.exists():
    rmtree(mlf_path)
mlf_path.mkdir()
for i, mlf in enumerate(cash_random.best_mlfs):
    joblib.dump(mlf, mlf_path / ("best_mlf_episode_%d.pkl" % (i + 1)))

cash_random.save(data_path / "rnn_metalearn_controller_experiment.pt")
