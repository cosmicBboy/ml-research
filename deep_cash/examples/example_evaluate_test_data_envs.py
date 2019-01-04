"""Evaluate controller after training."""
import pandas as pd
import os
import torch

from pathlib import Path
from sklearn.externals import joblib

from deep_cash.cash_controller import CASHController
from deep_cash.inference.inference_engine import CASHInference
from deep_cash.task_environment import TaskEnvironment
from deep_cash.data_environments import openml_api, sklearn_classification


build_path = Path(os.path.dirname(__file__)) / ".." / "floyd_outputs" / "225"

controller = CASHController.load(build_path / "controller_trial_0.pt")
experiment_results = pd.read_csv(
    build_path / "rnn_cash_controller_experiment.csv")
base_mlf_path = build_path / "cash_controller_mlfs_trial_0"


# get top 10 best mlfs for each data env across all episodes.
best_mlf_episodes = (
    experiment_results
    .groupby("data_env_names")
    .apply(lambda df: (
        df.sort_values("best_validation_scores", ascending=False).head(10)))
    ["episode"]
    .reset_index(level=1, drop=True)
)

# a dict mapping datasets to the top 10 mlfs found for those datasets.
best_mlfs = (
    best_mlf_episodes.map(
        lambda x: joblib.load(base_mlf_path / ("best_mlf_episode_%d.pkl" % x)))
    .groupby("data_env_names")
    .apply(lambda x: list(x))
    .to_dict()
)

sklearn_data_envs = sklearn_classification.envs()
openml_data_envs = openml_api.classification_envs()

torch.manual_seed(10)
task_env = TaskEnvironment(
    env_sources=["OPEN_ML", "SKLEARN"],
    test_set_config={"OPEN_ML": {"test_size": 0.8, "random_state": 100}},
    random_state=100,
    enforce_limits=True,
    per_framework_time_limit=720,
    per_framework_memory_limit=10000,
    dataset_names=list(sklearn_data_envs.keys()),
    test_dataset_names=list(openml_data_envs.keys()),
    error_reward=0,
    target_types=["BINARY", "MULTICLASS"])

inference_engine = CASHInference(controller, task_env)

# evaluate controller on test data environments
train_env_results = inference_engine.evaluate_training_data_envs(
    n=1, datasets=sklearn_data_envs.keys(), verbose=True)
test_env_results = inference_engine.evaluate_test_data_envs(n=50, verbose=True)
