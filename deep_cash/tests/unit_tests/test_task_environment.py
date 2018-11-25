"""Unit tests for task environment."""

import torch

from deep_cash.task_environment import TaskEnvironment


def test_task_env_datasampling():
    task_env = TaskEnvironment(
        env_sources=["SKLEARN"],
        random_state=100,
        per_framework_time_limit=10,
        per_framework_memory_limit=1000,
        error_reward=0)

    assert task_env.current_data_env is None
    assert task_env._current_task is None
    for _ in range(10):
        task_env.sample_data_distribution()
        assert task_env.current_data_env is not None
        for _ in range(10):
            metafeatures = task_env.sample_task()
            assert task_env._current_task is not None
            assert hasattr(task_env._current_task, "X_train")
            assert hasattr(task_env._current_task, "X_validation")
            assert hasattr(task_env._current_task, "y_train")
            assert hasattr(task_env._current_task, "y_validation")
            assert isinstance(metafeatures, torch.FloatTensor)
