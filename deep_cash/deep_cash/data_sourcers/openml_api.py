"""Source datasets from OpenML

https://www.openml.org/search?type=data
"""

import openml


DATASET_IDS = [
    2,  # anneal
]

DATA_CONFIG = [
    {"anneal": {
        "target_column": 38,
        "target_name": "class",
    }}
]


def get_datasets():
    results = openml.datasets.get_datasets(DATASET_IDS)
    return {r.name: r for r in results}


if __name__ == "__main__":
    D = get_datasets()
    dataset = D["anneal"]
    import ipdb; ipdb.set_trace()
