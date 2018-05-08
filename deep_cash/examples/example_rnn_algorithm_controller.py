"""Example usage of the Algorithm Controller."""

import matplotlib.pyplot as plt
import pandas as pd
import torch

from sklearn.metrics import roc_auc_score

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
num_candidates = 10

t_env = TaskEnvironment(
    roc_auc_score, random_state=100,
    per_framework_time_limit=5)

# create algorithm space
a_space = AlgorithmSpace()

n_layers = [3, 6]
metrics = pd.DataFrame(index=range(n_episodes))
best_frameworks = pd.DataFrame(index=range(num_candidates))
for n in n_layers:
    print("Training controller, n_layers=%d" % n)
    # create algorithm controller
    a_controller = AlgorithmControllerRNN(
        len(metafeatures), input_size=a_space.n_components,
        hidden_size=hidden_size, output_size=a_space.n_components,
        dropout_rate=0.3, num_rnn_layers=n)
    optim = torch.optim.Adam(a_controller.parameters(), lr=learning_rate)
    rewards, losses, ml_performances, best_candidates, best_scores = train(
        a_controller, a_space, t_env, optim, num_episodes=n_episodes,
        n_iter=n_iter, num_candidates=num_candidates)
    metrics["rewards_n_layers_%d" % n] = rewards
    metrics["losses_n_layers_%d" % n] = losses
    metrics["ml_performances_n_layers_%d" % n] = ml_performances
    best_frameworks["best_candidates_n_layers_%d" % n] = best_candidates
    best_frameworks["best_scores_n_layers_%d" % n] = best_scores
    print("\n")

for n in n_layers:
    print("Best model for n_layers=%d" % n)
    best_mlf = (
        best_frameworks
        .sort_values("best_scores_n_layers_%d" % n, ascending=False)
        ["best_candidates_n_layers_%d" % n].iloc[0])
    for step in best_mlf.steps:
        print(step)
    print("\n")

fig, ax = plt.subplots()
metrics[["rewards_n_layers_%d" % i for i in n_layers]].plot(ax=ax)
fig.savefig("artifacts/rnn_algorithm_controller_experiment.png")
metrics.to_csv(
    "artifacts/rnn_algorithm_controller_experiment.csv", index=False)
best_frameworks.to_csv(
    "artifacts/rnn_algorithm_controller_best_frameworks.csv", index=False)
