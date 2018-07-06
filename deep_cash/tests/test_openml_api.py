"""Unit and integration tests for Openml API."""

from deep_cash.data_sourcers import openml_api


def test_parse_dataset():
    datasets = [openml_api.parse_dataset(d) for d in openml_api.get_datasets()]
    assert len(datasets) == 1
    assert datasets[0]["dataset_name"] == "anneal"
    assert datasets[0]["data"].shape == (898, 38)  # 898 obs, 38 features
    assert datasets[0]["target"].shape == (898, )  # 898 obs, one class
