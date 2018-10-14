"""Source datasets from Kaggle API.

See https://github.com/Kaggle/kaggle-api for more details.
"""

import subprocess

from pathlib import Path

from . import kaggle_regression
from ..data_types import FeatureType, TargetType, DataSourceType


REGRESSION_COMPETITIONS = [
    # https://www.kaggle.com/c/restaurant-revenue-prediction
    "restaurant-revenue-prediction",
    # https://www.kaggle.com/c/nyc-taxi-trip-duration
    "nyc-taxi-trip-duration",
    # https://www.kaggle.com/c/mercedes-benz-greener-manufacturing
    "mercedes-benz-greener-manufacturing",
    # https://www.kaggle.com/c/allstate-claims-severity
    "allstate-claims-severity",
    # https://www.kaggle.com/c/house-prices-advanced-regression-techniques
    "house-prices-advanced-regression-techniques",
]
BINARY_CLASSIFICATION_COMPETITIONS = [
    # https://www.kaggle.com/c/homesite-quote-conversion
    "homesite-quote-conversion",
    # https://www.kaggle.com/c/santander-customer-satisfaction
    "santander-customer-satisfaction",
    # https://www.kaggle.com/c/bnp-paribas-cardif-claims-management
    "bnp-paribas-cardif-claims-management",
]
MULTI_CLASSIFICATION_COMPETITIONS = [
    # https://www.kaggle.com/c/poker-rule-induction
    "poker-rule-induction",
    # https://www.kaggle.com/c/costa-rican-household-poverty-prediction
    "costa-rican-household-poverty-prediction",
]


def classification_envs():
    return []


def regression_envs():
    return [
        kaggle_regression.restaurant_revenue_prediction().create_data_env(),
        kaggle_regression.nyc_taxi_trip_duration().create_data_env(),
    ]


def envs():
    return classification_envs() + regression_envs()
