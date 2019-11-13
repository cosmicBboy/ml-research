"""Tests for benchmark dataset suites."""

import numpy as np
import pytest

from metalearn.task_environment import TaskEnvironment


@pytest.mark.parametrize(
    "env_source", ["AUTOSKLEARN_BENCHMARK", "OPEN_ML_BENCHMARK_CC18"]
)
def test_load_autosklearn_task_environment(env_source):
    task_env = TaskEnvironment(
        env_sources=[env_source],
        target_types=[
            "BINARY",
            "MULTICLASS"
        ],
        random_state=100,
        enforce_limits=True,
        per_framework_time_limit=10,
        per_framework_memory_limit=1000,
        dataset_names=None,
        error_reward=0,
        n_samples=500)
    for data_env in task_env.data_distribution:
        task_env.set_data_env(data_env)
        for _ in range(3):
            task_env.sample_task_state()
            assert isinstance(task_env._current_task.X_train, np.ndarray)
            assert isinstance(
                task_env._current_task.X_validation, np.ndarray)
            assert isinstance(task_env._current_task.y_train, np.ndarray)
            assert isinstance(
                task_env._current_task.y_validation, np.ndarray)
