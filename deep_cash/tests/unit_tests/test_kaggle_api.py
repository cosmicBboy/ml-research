"""Unit tests for Kaggle API."""

from deep_cash.data_types import TargetType, DataSourceType
from deep_cash.data_environments import kaggle_regression, \
    kaggle_classification


def test_kaggle_regression_competitions():
    kaggle_comps = [
        kaggle_regression.restaurant_revenue_prediction(),
        kaggle_regression.nyc_taxi_trip_duration(),
        kaggle_regression.mercedes_benz_greener_manufacturing(),
        kaggle_regression.allstate_claims_severity(),
        kaggle_regression.house_prices_advanced_regression_techniques(),
    ]
    for comp in kaggle_comps:
        env = comp.data_env()
        assert env.target_type == TargetType.REGRESSION
        assert env.source == DataSourceType.KAGGLE


def test_kaggle_classification_competitions():
    kaggle_comps = [
        kaggle_classification.homesite_quote_conversion(),
        kaggle_classification.santander_customer_satisfaction(),
        kaggle_classification.bnp_paribas_cardif_claims_management(),
        kaggle_classification.poker_rule_induction(),
        kaggle_classification.costa_rican_household_poverty_prediction(),
    ]
    for comp in kaggle_comps:
        env = comp.data_env()
        assert env.target_type in [TargetType.BINARY, TargetType.MULTICLASS]
        assert env.source == DataSourceType.KAGGLE
