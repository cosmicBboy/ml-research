"""Unit tests for task environment."""

from deep_cash.task_environment import TaskEnvironment


def test_task_env_datasampling():
    return TaskEnvironment(
        random_state=100,
        per_framework_time_limit=10,
        per_framework_memory_limit=1000,
        error_reward=0)
