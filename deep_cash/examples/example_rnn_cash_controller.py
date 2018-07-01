"""Example usage of the CASH Controller.

# TODO: experiment with a different set of reward schemes,
# e.g. -1 for error during fit, 100 scale for model validation performance.
"""

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
from deep_cash.loggers import get_loggers
from deep_cash.utils import METAFEATURE_DIM


data_path = Path(os.environ.get(
    "DEEP_CASH_OUT_PATH", os.path.dirname(__file__) + "/artifacts"))

# hyperparameters
n_episodes = int(os.environ.get("DEEP_CASH_N_EPISODES", 500))
n_iter = int(os.environ.get("DEEP_CASH_N_ITER", 50))
learning_rate = float(os.environ.get("DEEP_CASH_LEARNING_RATE", 0.005))
error_reward = int(os.environ.get("DEEP_CASH_ERROR_REWARD", -1))
logger_name = os.environ.get("DEEP_CASH_LOGGER", None)
logger = get_loggers().get(os.environ.get("DEEP_CASH_LOGGER", None), None)
fit_verbose = int(os.environ.get("DEEP_CASH_FIT_VERBOSE", 1))

hidden_size = 30
output_size = 30
n_layers = 3


t_env = TaskEnvironment(
    f1_score,
    scorer_kwargs={"average": "weighted"},
    random_state=100,
    per_framework_time_limit=60,
    per_framework_memory_limit=3077,
    error_reward=error_reward,
    reward_transformer=lambda x: x)

# create algorithm space
a_space = AlgorithmSpace(
    with_end_token=False,
    hyperparam_with_start_token=False,
    hyperparam_with_none_token=False)

controller = CASHController(
    metafeature_size=METAFEATURE_DIM,
    input_size=a_space.n_components,
    hidden_size=hidden_size,
    output_size=output_size,
    a_space=a_space,
    optim=torch.optim.Adam,
    optim_kwargs={"lr": learning_rate},
    dropout_rate=0.2,
    num_rnn_layers=n_layers)

reinforce = CASHReinforce(controller, t_env, metrics_logger=logger)
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
]].to_csv(str(data_path / "rnn_cash_controller_experiment.csv"), index=False)

mlf_path = data_path / "rnn_cash_controller_experiment_mlfs"
if mlf_path.exists():
    rmtree(mlf_path)
mlf_path.mkdir()
for i, mlf in enumerate(history.best_mlfs):
    joblib.dump(mlf, mlf_path / ("best_mlf_episode_%d.pkl" % (i + 1)))

controller.save(data_path / "rnn_cash_controller_experiment.pt")