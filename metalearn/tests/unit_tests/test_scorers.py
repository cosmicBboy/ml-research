"""Unit tests for scorer objects."""

import pytest

from metalearn import scorers


def test_exponentiated_log():
    assert scorers.exponentiated_log(0) == 1
    assert scorers.exponentiated_log(10) < scorers.exponentiated_log(1)
    assert scorers.exponentiated_log(10, gamma=10) < \
        scorers.exponentiated_log(10, gamma=0.01)

    for i in [-0.01, -1, -10]:
        with pytest.raises(ValueError):
            assert scorers.exponentiated_log(i)


def test_rectified_linear():
    assert scorers.rectified_linear(0) == 0

    for i in [1, 10, 18, 90]:
        assert scorers.rectified_linear(i) == i

    for i in [-1, -0.5, -100]:
        assert scorers.rectified_linear(i) == 0


def test_classification_scorers():
    y_pred = [1, 0, 0, 0, 1, 1]
    y_true = [1, 1, 1, 0, 0, 0]

    for scorer in [
            scorers.accuracy(),
            scorers.precision(),
            scorers.recall(),
            scorers.f1_score_weighted_average(),
            scorers.f1_score_macro(),
            scorers.log_loss(),
            scorers.roc_auc()]:
        score = scorer.fn(y_true, y_pred)
        reward = scorer.reward_transformer(score) if \
            scorer.reward_transformer else score
        # rewards should be calibrated to be between 0 and 1
        assert 0 <= reward <= 1

        best_score = scorer.fn(y_true, y_true)
        best_reward = scorer.reward_transformer(best_score) if \
            scorer.reward_transformer else best_score
        assert best_reward == 1
        assert best_reward > reward
        # should return True if first score is better than second
        assert scorer.comparator(best_score, score)


def test_regression_scorers():
    y_pred = [11, 22, 28, 50]
    y_true = [10, 20, 30, 40]

    for scorer in [
            scorers.mean_absolute_error(),
            scorers.mean_squared_error(),
            scorers.r2_score(),
            scorers.root_mean_squared_error(),
            scorers.root_mean_squared_log_error()]:
        score = scorer.fn(y_true, y_pred)
        reward = scorer.reward_transformer(score) if \
            scorer.reward_transformer else score
        assert 0 <= reward <= 1

        best_score = scorer.fn(y_true, y_true)
        best_reward = scorer.reward_transformer(best_score) if \
            scorer.reward_transformer else best_score
        assert best_reward == 1
        assert best_reward > reward
        # should return True if first score is better than second
        assert scorer.comparator(best_score, score)
