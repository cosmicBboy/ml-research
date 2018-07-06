"""Unit and integration tests for Openml API."""

from deep_cash.data_sourcers import openml_api


def test_parse_dataset():
    datasets = [openml_api.parse_dataset(d) for d in openml_api.get_datasets()]
    assert len(datasets) == 1
    assert datasets[0]["dataset_name"] == "anneal"
    assert datasets[0]["data"].shape == (898, 38)  # 898 obs, 38 features
    assert datasets[0]["target"].shape == (898, )  # 898 obs, one class


def test_list_classification_datasets():
    dataset_metadata = openml_api.list_classification_datasets(n_results=10)
    assert len(dataset_metadata) == 10
    for metadata in dataset_metadata.values():
        assert metadata["NumberOfMissingValues"] == 0
        assert openml_api.CLASS_RANGE[0] <= metadata["NumberOfClasses"] <= \
            openml_api.CLASS_RANGE[1]


def test_classification_envs():
    datasets = openml_api.classification_envs()
    # the 10 default classification datasets + 1 custom dataset
    assert len(datasets) == 11
