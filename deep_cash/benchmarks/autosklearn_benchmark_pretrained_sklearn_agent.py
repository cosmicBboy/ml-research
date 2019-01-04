"""Benchmark of the autosklearn datasets using pre-trained baseline agent.

This agent is pre-trained on the following sklearn datasets:
- iris
- digits
- wine
- breast_cancer
"""

import pandas as pd
import os
import torch

from pathlib import Path
from sklearn.externals import joblib

from deep_cash.cash_controller import CASHController
from deep_cash.inference.inference_engine import CASHInference
from deep_cash.task_environment import TaskEnvironment
from deep_cash.data_environments import sklearn_classification


build_path = Path(os.path.dirname(__file__)) / ".." / "floyd_outputs" / "225"
output_path = Path(os.path.dirname(__file__)) / "results" / \
    "autosklearn_benchmark_pretrained_sklearn_agent"
results_path = output_path / "inference_results"

results_path.mkdir(exist_ok=True)

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
torch.manual_seed(10)
task_env = TaskEnvironment(
    env_sources=["SKLEARN"],
    test_env_sources=["AUTOSKLEARN_BENCHMARK"],
    test_set_config={
        "AUTOSKLEARN_BENCHMARK": {
            "test_size": 0.8, "random_state": 100, "verbose": 5}
    },
    random_state=100,
    enforce_limits=True,
    per_framework_time_limit=720,
    per_framework_memory_limit=10000,
    dataset_names=list(sklearn_data_envs.keys()),
    test_dataset_names=None,
    error_reward=0,
    target_types=["BINARY", "MULTICLASS"])

inference_engine = CASHInference(controller, task_env)


def create_inference_result_df(inference_results):
    """Create """
    def _tabulate_results(fn, key):
        df = pd.DataFrame(
            {k: [fn(x) for x in v]
             for k, v in inference_results.items()})
        df = df.unstack().reset_index([0, 1])
        df.columns = ["data_env", "n_inference_steps", "value"]
        df["key"] = key
        return df[["data_env", "n_inference_steps", "key", "value"]]

    def _tabulate_mlf(mlf):
        if mlf is None:
            return mlf
        else:
            return str(mlf.named_steps)

    return pd.concat([
        _tabulate_results(lambda x: x.validation_score, "validation_score"),
        _tabulate_results(lambda x: x.reward, "reward"),
        _tabulate_results(lambda x: x.is_valid, "is_valid"),
        _tabulate_results(lambda x: x.mlf_description, "mlf_description")
    ])


# evaluate controller on training data envs
train_env_inference_results = inference_engine.evaluate_training_data_envs(
    n=500, datasets=sklearn_data_envs.keys(), verbose=True)
train_env_results_df = create_inference_result_df(train_env_inference_results)
train_env_results_df.to_csv(
    results_path / "train_env_inference_results.csv", index=False)

# evaluate controller on test data environments
for test_data_env in task_env.test_data_distribution:
    test_env_inference_results = inference_engine.evaluate_test_data_envs(
        n=500, datasets=[test_data_env.name], verbose=True)
    test_env_results_df = create_inference_result_df(
        test_env_inference_results)
    test_env_results_df.to_csv(
        results_path / ("test_env_inference_results_%s.csv" %
                        test_data_env.name),
        index=False)
