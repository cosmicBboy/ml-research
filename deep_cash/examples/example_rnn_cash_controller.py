"""Example usage of the CASH Controller."""

import os
import pandas as pd
import torch

from pathlib import Path
from shutil import rmtree
from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score, f1_score

from deep_cash.cash_controller import CASHController
from deep_cash.cash_reinforce import CASHReinforce

from deep_cash.algorithm_space import AlgorithmSpace
from deep_cash.task_environment import TaskEnvironment


data_path = Path(os.path.dirname(__file__)) / "artifacts"

# hyperparameters
metafeatures = ["number_of_examples"]
learning_rate = 0.005
hidden_size = 30
output_size = 30
n_layers = 3


t_env = TaskEnvironment(
    f1_score, scorer_kwargs={"average": "weighted"}, random_state=100,
    per_framework_time_limit=5)

# create algorithm space
a_space = AlgorithmSpace(
    with_end_token=False,
    hyperparam_with_start_token=False,
    hyperparam_with_none_token=False)

controller = CASHController(
    metafeature_size=len(metafeatures),
    input_size=a_space.n_components,
    hidden_size=hidden_size,
    output_size=output_size,
    a_space=a_space,
    optim=torch.optim.Adam,
    optim_kwargs={"lr": learning_rate},
    dropout_rate=0.2,
    num_rnn_layers=n_layers)

reinforce = CASHReinforce(controller, t_env)
reinforce.fit(n_episodes=1000, n_iter=100)


history = pd.DataFrame(reinforce.history())
history[[
    "episode",
    "data_env_names",
    "losses",
    "mean_validation_scores",
    "std_validation_scores",
    "n_successful_mlfs",
    "n_unique_mlfs",
    "best_validation_scores",
]].to_csv(str(data_path / "rnn_cash_controller_experiment.csv"), index=False)

mlf_path = data_path / "rnn_cash_controller_experiment_mlfs"
rmtree(mlf_path)
mlf_path.mkdir()
for i, mlf in enumerate(history.best_mlfs):
    joblib.dump(mlf, mlf_path / ("best_mlf_episode_%d.pkl" % (i + 1)))
