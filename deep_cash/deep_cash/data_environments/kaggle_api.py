"""Source datasets from Kaggle API.

See https://github.com/Kaggle/kaggle-api for more details.
"""

from collections import ChainMap, OrderedDict

from . import kaggle_regression, kaggle_classification


def classification_envs():
    competitions = [
        kaggle_classification.homesite_quote_conversion(),
        kaggle_classification.santander_customer_satisfaction(),
        kaggle_classification.bnp_paribas_cardif_claims_management(),
        kaggle_classification.poker_rule_induction(),
        kaggle_classification.costa_rican_household_poverty_prediction(),
    ]
    return OrderedDict([(d.dataset_name, d.data_env()) for d in competitions])


def regression_envs():
    competitions = [
        kaggle_regression.restaurant_revenue_prediction(),
        kaggle_regression.nyc_taxi_trip_duration(),
        kaggle_regression.mercedes_benz_greener_manufacturing(),
        kaggle_regression.allstate_claims_severity(),
        kaggle_regression.house_prices_advanced_regression_techniques(),
    ]
    return OrderedDict([(d.dataset_name, d.data_env()) for d in competitions])


def envs(dataset_names):
    envs = ChainMap(classification_envs(), regression_envs())
    if dataset_names is None:
        return [envs[d] for d in envs]
    return [envs[d] for d in dataset_names]
