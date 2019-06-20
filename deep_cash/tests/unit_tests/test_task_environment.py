"""Unit tests for task environment."""

import torch

from metalearn.task_environment import TaskEnvironment


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
        task_env.sample_data_env()
        assert task_env.current_data_env is not None
        for _ in range(10):
            metafeatures = task_env.sample_task_state()
            assert task_env._current_task is not None
            assert hasattr(task_env._current_task, "X_train")
            assert hasattr(task_env._current_task, "X_validation")
            assert hasattr(task_env._current_task, "y_train")
            assert hasattr(task_env._current_task, "y_validation")
            assert isinstance(metafeatures, torch.FloatTensor)


def test_task_env_scorer_metafeature():
    for target_type in ["BINARY", "MULTICLASS", "REGRESSION"]:
        task_env = TaskEnvironment(
            env_sources=["SKLEARN"],
            target_types=[target_type],
            include_scoring_metafeature=True,
            random_state=100,
            error_reward=0)
        scorer_distribution = task_env.target_type_to_scorer_distribution[
            task_env.target_types[0]]

        assert task_env._include_scoring_metafeature
        assert [target_type] == [
            t.name for t in task_env.target_type_to_scorer_distribution.keys()]
        assert any(map(
            lambda x: "scorer_distribution" in x, task_env.metafeature_spec))

        n_X_features = 2  # number of feature derived from X
        n_data_envs = task_env.n_data_envs + 1  # include the NULL data env
        sampled_scorer_names = []
        for _ in range(50):
            task_env.sample_data_env()
            sampled_scorer_names.append(task_env.scorer.name)
            for _ in range(10):
                metafeatures = task_env.sample_task_state()
                assert metafeatures.view(-1).shape[0] == \
                    n_X_features + n_data_envs + len(scorer_distribution)
            assert task_env.scorer in scorer_distribution
        assert len(set(sampled_scorer_names)) > 1
