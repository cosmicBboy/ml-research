"""Example usage of the Algorithm Controller."""

import matplotlib.pyplot as plt
import pandas as pd
import torch

from sklearn.metrics import roc_auc_score, f1_score

from deep_cash.algorithm_space import AlgorithmSpace
from deep_cash.rnn_algorithm_controller import (
    AlgorithmControllerRNN, HyperparameterControllerRNN)
from deep_cash.rnn_ml_framework_controller import MLFrameworkController
from deep_cash.task_environment import TaskEnvironment
from deep_cash.utils import load_model


# a single metafeature about the dataset-task.
# TODO: add metafeatures re: supervised task
metafeatures = ["number_of_examples"]
learning_rate = 0.005
hidden_size = 100
n_episodes = 10
activate_h_controller = 1000
n_iter = 1000
num_candidates = 10
sig_check_interval = 50

t_env = TaskEnvironment(
    f1_score, scorer_kwargs={"average": "weighted"}, random_state=100,
    per_framework_time_limit=5)

# create algorithm space
a_space = AlgorithmSpace(with_end_token=False)

n_layers = 3
metrics = pd.DataFrame(index=range(n_episodes))
best_frameworks = pd.DataFrame(index=range(num_candidates))

print("Training controller, n_layers=%d" % n_layers)

# try create algorithm controller from pretrained algorithm
try:
    a_controller = load_model(
        "artifacts/pretrained_rnn_algorithm_controller.pt",
        AlgorithmControllerRNN, len(metafeatures),
        input_size=a_space.n_components, hidden_size=hidden_size,
        output_size=a_space.n_components, dropout_rate=0.3,
        num_rnn_layers=n_layers)
except Exception:
    a_controller = AlgorithmControllerRNN(
        len(metafeatures), input_size=a_space.n_components,
        hidden_size=hidden_size, output_size=a_space.n_components,
        dropout_rate=0.3, num_rnn_layers=n_layers)

h_controller = HyperparameterControllerRNN(
    len(metafeatures) + (a_space.n_components * a_space.N_COMPONENT_TYPES),
    input_size=a_space.n_hyperparameters,
    hidden_size=hidden_size, output_size=a_space.n_hyperparameters,
    dropout_rate=0.3, num_rnn_layers=n_layers)
mlf_controller = MLFrameworkController(
    a_controller, h_controller, a_space,
    optim=torch.optim.Adam, optim_kwargs={"lr": learning_rate})

tracker = mlf_controller.fit(
    t_env, num_episodes=n_episodes, n_iter=n_iter,
    num_candidates=num_candidates,
    activate_h_controller=activate_h_controller,
    increase_n_hyperparam_by=1, increase_n_hyperparam_every=10,
    sig_check_interval=sig_check_interval)

best_candidates = tracker.best_candidates + \
    [None] * (num_candidates - len(tracker.best_candidates))
best_scores = tracker.best_scores + \
    [None] * (num_candidates - len(tracker.best_scores))

# gather metrics
metrics["rewards_n_layers_%d" % n_layers] = tracker.overall_mean_reward
metrics["algorithm_losses_n_layers_%d" % n_layers] = tracker.overall_a_loss
metrics["hyperparam_losses_n_layers_%d" % n_layers] = tracker.overall_h_loss
metrics["ml_score_n_layers_%d" % n_layers] = tracker.overall_ml_score
best_frameworks["best_cand_n_layers_%d" % n_layers] = best_candidates
best_frameworks["best_scores_n_layers_%d" % n_layers] = best_scores
print("\n")

print("Best model for n_layers=%d" % n_layers)
best_mlf = (
    best_frameworks
    .sort_values("best_scores_n_layers_%d" % n_layers, ascending=False)
    ["best_cand_n_layers_%d" % n_layers].iloc[0])
for step in best_mlf.steps:
    print(step)
print("\n")

fig, ax = plt.subplots()
metrics[["rewards_n_layers_%d" % i for i in [n_layers]]].plot(ax=ax)
fig.savefig("artifacts/rnn_algorithm_controller_experiment.png")
metrics.to_csv(
    "artifacts/rnn_algorithm_controller_experiment.csv", index=False)
best_frameworks.to_csv(
    "artifacts/rnn_algorithm_controller_best.csv", index=False)
# torch.save(a_controller.state_dict(),
#            "artifacts/pretrained_rnn_algorithm_controller.pt")
torch.save(h_controller.state_dict(), "artifacts/rnn_hyperparam_controller.pt")
