"""Module for evaluating trained controller."""

import os

from pathlib import Path

import pandas as pd
import torch

from metalearn.metalearn_controller import MetaLearnController
from metalearn.inference.inference_engine import CASHInference
from metalearn.task_environment import TaskEnvironment


MODEL_PATH = Path(os.path.dirname(__file__)) / ".." / "floyd_outputs" / "290"
OUTPUT_PATH = Path.home() / ".metalearn-cache" / "inference_results" / "290"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


def load_experiment_results():
    return pd.read_csv(
        MODEL_PATH / "rnn_metalearn_controller_experiment.csv")


def load_controller():
    return MetaLearnController.load(MODEL_PATH / "controller_trial_0.pt")


def load_eval_task_env():
    torch.manual_seed(10)
    return TaskEnvironment(
        env_sources=["SKLEARN", "OPEN_ML", "KAGGLE"],
        test_env_sources=["AUTOSKLEARN_BENCHMARK"],
        test_set_config={
            "AUTOSKLEARN_BENCHMARK": {
                "test_size": 0.8, "random_state": 100, "verbose": 5}
        },
        random_state=100,
        enforce_limits=True,
        per_framework_time_limit=720,
        per_framework_memory_limit=10000,
        dataset_names=None,
        test_dataset_names=None,
        error_reward=0,
        target_types=["BINARY"])


def _compute_inference_results(
        task_env, inference_engine, n_shots, n_eval_samples,
        evaluate_test_env=False):

    if evaluate_test_env:
        data_distribution = task_env.test_data_distribution
        inference_fn = inference_engine.evaluate_test_data_envs
    else:
        data_distribution = task_env.data_distribution
        inference_fn = inference_engine.evaluate_training_data_envs

    data_dist_inference_results = []
    for data_env in data_distribution:
        for i in range(n_eval_samples):
            inference_results = inference_fn(
                n=n_shots, datasets=[data_env.name], verbose=True)
            data_dist_inference_results.append(
                create_inference_result_df(inference_results)
                .assign(
                    data_env=data_env.name,
                    target_type=data_env.target_type.name,
                    n_eval_sample=i,
                )
            )
    data_dist_inference_results = \
        None if len(data_dist_inference_results) == 0 \
        else pd.concat(data_dist_inference_results)

    return data_dist_inference_results


def evaluate_controller(
        controller, task_env, meta_reward_multiplier, n_shots=20,
        n_eval_samples=5):

    inference_engine = CASHInference(
        controller, task_env, meta_reward_multiplier)

    # evaluate controller on training data envs
    train_env_inference_results = _compute_inference_results(
        task_env, inference_engine, n_shots, n_eval_samples,
    )

    # evaluate controller on test data environments
    test_env_inference_results = _compute_inference_results(
        task_env, inference_engine, n_shots, n_eval_samples,
        evaluate_test_env=True,
    )

    return {
        "training_env": train_env_inference_results,
        "test_env": test_env_inference_results,
    }


def create_inference_result_df(inference_results):

    def _tabulate_results(fn, key):
        df = pd.DataFrame({
            k: [fn(x) for x in v]
            for k, v in inference_results.items()
        })
        df = df.unstack().reset_index([0, 1])
        df.columns = ["data_env", "n_inference_steps", "value"]
        df["key"] = key
        return df[["data_env", "n_inference_steps", "key", "value"]]

    return pd.concat([
        _tabulate_results(lambda x: x.validation_score, "validation_score"),
        _tabulate_results(lambda x: x.reward, "reward"),
        _tabulate_results(lambda x: x.is_valid, "is_valid"),
        _tabulate_results(lambda x: x.mlf, "mlf"),
        _tabulate_results(lambda x: x.mlf_full, "mlf_full"),
        _tabulate_results(lambda x: x.scorer, "scorer"),
        _tabulate_results(lambda x: x.target_type, "target_type"),
    ])
