"""Unit tests for reinforce module."""

import torch

from metalearn import metalearn_reinforce


def test_normalize_reward_happy_path():
    """Test happy path for normalize_reward function."""
    reward_buffer = torch.FloatTensor([1, 3, 5])
    expected = torch.FloatTensor([-1, 0, 1])
    result = metalearn_reinforce.normalize_reward(reward_buffer)
    assert all(result == expected)


def test_normalize_reward_zero_std():
    """Test that normalize_reward can handle 0 standard deviation."""
    reward_buffer = torch.FloatTensor([5, 5, 5])
    expected = torch.FloatTensor([0, 0, 0])
    result = metalearn_reinforce.normalize_reward(reward_buffer)
    assert all(result == expected)
