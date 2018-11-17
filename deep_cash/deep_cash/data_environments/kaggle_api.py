"""Source datasets from Kaggle API.

See https://github.com/Kaggle/kaggle-api for more details.
"""

import subprocess

from . import kaggle_regression, kaggle_classification


def classification_envs():
    return [
        kaggle_classification.homesite_quote_conversion().data_env(),
        kaggle_classification.santander_customer_satisfaction().data_env(),
        kaggle_classification.bnp_paribas_cardif_claims_management().data_env(),  # noqa E501
        kaggle_classification.poker_rule_induction().data_env(),
        kaggle_classification.costa_rican_household_poverty_prediction().data_env(),  # noqa E501
    ]


def regression_envs():
    return [
        kaggle_regression.restaurant_revenue_prediction().data_env(),
        kaggle_regression.nyc_taxi_trip_duration().data_env(),
        kaggle_regression.mercedes_benz_greener_manufacturing().data_env(),
        kaggle_regression.allstate_claims_severity().data_env(),
        kaggle_regression.house_prices_advanced_regression_techniques().data_env()  # noqa E501
    ]


def envs():
    return classification_envs() + regression_envs()
