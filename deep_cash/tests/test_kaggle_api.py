"""Unit tests for Kaggle API."""

from deep_cash.data_types import TargetType, DataSourceType
from deep_cash.data_sourcers import kaggle_regression


def test_kaggle_regression_competitions():
    kaggle_comps = [
        kaggle_regression.restaurant_revenue_prediction(),
        kaggle_regression.nyc_taxi_trip_duration(),
        kaggle_regression.mercedes_benz_greener_manufacturing(),
        kaggle_regression.allstate_claims_severity(),
        kaggle_regression.house_prices_advanced_regression_techniques(),
    ]
    for comp in kaggle_comps:
        env = comp.create_data_env()
        assert env["target_type"] == TargetType.REGRESSION
        assert env["source"] == DataSourceType.KAGGLE
