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


def evaluate_controller(controller, task_env, n=20):
    inference_engine = CASHInference(controller, task_env)

    # evaluate controller on training data envs
    train_env_inference_results = []
    for train_data_env in task_env.data_distribution:
        inference_results = inference_engine.evaluate_training_data_envs(
            n=n, datasets=[train_data_env.name], verbose=True)
        train_env_inference_results.append(
            create_inference_result_df(inference_results)
            .assign(data_env=train_data_env.name))
    train_env_inference_results = \
        None if len(train_env_inference_results) == 0 \
        else pd.concat(train_env_inference_results)

    # evaluate controller on test data environments
    test_env_inference_results = []
    for test_data_env in task_env.test_data_distribution:
        inference_results = inference_engine.evaluate_test_data_envs(
            n=n, datasets=[test_data_env.name], verbose=True)
        test_env_inference_results.append(
            create_inference_result_df(inference_results)
            .assign(data_env=test_data_env.name))
    test_env_inference_results = None if len(test_env_inference_results) == 0 \
        else pd.concat(test_env_inference_results)

    return {
        "training_env": train_env_inference_results,
        "test_env": test_env_inference_results,
    }


def create_inference_result_df(inference_results):
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
        _tabulate_results(lambda x: x.mlf, "mlf"),
        _tabulate_results(lambda x: x.mlf_full, "mlf_full"),
        _tabulate_results(lambda x: x.scorer, "scorer"),
        _tabulate_results(lambda x: x.target_type, "target_type"),
    ])
