"""Example usage of the CASH Controller."""

import torch

from sklearn.metrics import roc_auc_score, f1_score

from deep_cash.cash_controller import CASHController

from deep_cash.algorithm_space import AlgorithmSpace
from deep_cash.task_environment import TaskEnvironment


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
