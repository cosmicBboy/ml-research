"""Example usage of the Algorithm Controller."""

import matplotlib.pyplot as plt
import pandas as pd
import torch

from sklearn.metrics import roc_auc_score

from deep_cash import components
from deep_cash.algorithm_space import AlgorithmSpace
from deep_cash.rnn_algorithm_controller import AlgorithmControllerRNN, train
from deep_cash.task_environment import TaskEnvironment


# a single metafeature about the dataset-task.
# TODO: add metafeatures re: supervised task
metafeatures = ["number_of_examples"]
learning_rate = 0.005
hidden_size = 100
n_episodes = 100
n_iter = 100

t_env = TaskEnvironment(roc_auc_score, random_state=100)

# create algorithm space
a_space = AlgorithmSpace()

df = pd.DataFrame(index=range(n_episodes))
n_layers = [3, 6]
for n in n_layers:
    print("Training controller, n_layers=%d" % n)
    # create algorithm controller
    a_controller = AlgorithmControllerRNN(
        len(metafeatures), input_size=a_space.n_components,
        hidden_size=hidden_size, output_size=a_space.n_components,
        dropout_rate=0.3, num_rnn_layers=n)
    optim = torch.optim.Adam(a_controller.parameters(), lr=learning_rate)
    rewards, losses = train(
        a_controller, a_space, t_env, optim, num_episodes=n_episodes,
        n_iter=n_iter)
    df["rewards_n_layers_%d" % n] = rewards
    df["losses_n_layers_%d" % n] = losses
    print("\n")

fig, ax = plt.subplots()
df[["rewards_n_layers_%d" % i for i in n_layers]].plot(ax=ax)
fig.savefig("artifacts/rnn_algorithm_controller_experiment.png")
df.to_csv("artifacts/rnn_algorithm_controller_experiment.csv", index=False)

