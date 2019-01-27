"""Evaluate the test sets of cash controller."""

# TODO: this script is still under construction!

import pandas as pd
import os
import torch

from collections import OrderedDict
from pathlib import Path
from sklearn.externals import joblib

from deep_cash.cash_controller import CASHController
from deep_cash.inference.inference_engine import CASHInference
from deep_cash.task_environment import TaskEnvironment
from deep_cash.data_environments import sklearn_classification


build_path = Path(os.path.dirname(__file__)) / ".." / "floyd_outputs" / "225"

controller = CASHController.load(build_path / "controller_trial_0.pt")
experiment_results = pd.read_csv(
    build_path / "rnn_cash_controller_experiment.csv")
base_mlf_path = build_path / "cash_controller_mlfs_trial_0"

# will need to make this part more elegant. But for now assume that we know
# each dataset's corresponding scorer. Ideally, which scorer was used for a
# particular episode would just be metadata in the
# rnn_cash_controller_experiment.csv file.
sklearn_dataenvs = sklearn_classification.envs()

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

# modify this controller so that it supports the NULL data env. This shouldn't
# be necessary in the future.
torch.manual_seed(10)
task_env = TaskEnvironment(
    env_sources=["SKLEARN"],
    test_set_config={"SKLEARN": {"test_size": 0.8, "random_state": 100}},
    random_state=100,
    enforce_limits=True,
    per_framework_time_limit=720,
    per_framework_memory_limit=10000,
    dataset_names=...,
    test_dataset_names=["sklearn.boston"],
    error_reward=0,
    target_types=["BINARY", "MULTICLASS"])

inference_engine = CASHInference(controller, task_env)

# evaluate controller on the test set of the training data environment
# distribution.
test_set_results = OrderedDict()
for dataset, mlfs in best_mlfs.items():
    test_set_results[dataset] = inference_engine.evaluate_test_sets(
        dataset, mlfs)
