"""Example usage of the Algorithm Controller."""

from sklearn.metrics import roc_auc_score
import torch

from deep_cash import components
from deep_cash.algorithm_space import AlgorithmSpace
from deep_cash.rnn_algorithm_controller import AlgorithmControllerRNN, train
from deep_cash.task_environment import TaskEnvironment


# a single metafeature about the dataset-task.
# TODO: add metafeatures re: supervised task
metafeatures = ["number_of_examples"]
hidden_size = 10
learning_rate = 0.0005

t_env = TaskEnvironment(roc_auc_score)

# create algorithm space
a_space = AlgorithmSpace(
    data_preprocessors=[components.data_preprocessors.imputer()],
    feature_preprocessors=[components.feature_preprocessors.pca()],
    classifiers=[components.classifiers.logistic_regression()])

# create algorithm controller
a_controller = AlgorithmControllerRNN(
    len(metafeatures), input_size=a_space.n_components,
    hidden_size=hidden_size, output_size=a_space.n_components)

optim = torch.optim.Adam(a_controller.parameters(), lr=learning_rate)

train(a_controller, a_space, t_env, optim)
