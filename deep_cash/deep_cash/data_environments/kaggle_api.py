"""Source datasets from Kaggle API.

See https://github.com/Kaggle/kaggle-api for more details.
"""

from collections import OrderedDict

from . import kaggle_regression, kaggle_classification


def classification_envs(n=5, test_size=None, random_state=None, verbose=False):
    # TODO: limit number of envs by n
    competitions = [
        kaggle_classification.homesite_quote_conversion(),
        kaggle_classification.santander_customer_satisfaction(),
        kaggle_classification.bnp_paribas_cardif_claims_management(),
        kaggle_classification.poker_rule_induction(),
        kaggle_classification.costa_rican_household_poverty_prediction(),
    ]
    return OrderedDict([
        (d.dataset_name, d.data_env(
            test_size=test_size, random_state=random_state))
        for d in competitions])


def regression_envs(n=5, test_size=None, random_state=None, verbose=False):
    # TODO: limit number of envs by n
    competitions = [
        kaggle_regression.restaurant_revenue_prediction(),
        kaggle_regression.nyc_taxi_trip_duration(),
        kaggle_regression.mercedes_benz_greener_manufacturing(),
        kaggle_regression.allstate_claims_severity(),
        kaggle_regression.house_prices_advanced_regression_techniques(),
    ]
    return OrderedDict([
        (d.dataset_name, d.data_env(
            test_size=test_size, random_state=random_state))
        for d in competitions])
