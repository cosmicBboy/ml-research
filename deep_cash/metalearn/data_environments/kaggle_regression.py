"""Metadata for kaggle regression datasets."""

from collections import OrderedDict

from .kaggle_base import KaggleCompetition

from ..data_types import FeatureType, TargetType
from .. import scorers


def restaurant_revenue_prediction():
    """Create data interface to kaggle 'restaurant revenue prediction'.

    url: https://www.kaggle.com/c/restaurant-revenue-prediction
    """
    return KaggleCompetition(
        competition_id="restaurant-revenue-prediction",
        features=OrderedDict(
            [("Open Date", FeatureType.DATE),
             ("City", FeatureType.CATEGORICAL),
             ("City Group", FeatureType.CATEGORICAL),
             ("Type", FeatureType.CATEGORICAL)] +
            [("P%i" % (i + 1), FeatureType.CONTINUOUS) for i in range(37)]
        ),
        target={"revenue": TargetType.REGRESSION},
        training_data_fname="train.csv.zip",
        test_data_fname="test.csv.zip",
        custom_preprocessor=None,
        scorer=scorers.root_mean_squared_error())


def nyc_taxi_trip_duration():
    """Create data interface to kaggle 'nyc taxi trip duration'.

    url: https://www.kaggle.com/c/nyc-taxi-trip-duration
    """
    return KaggleCompetition(
        competition_id="nyc-taxi-trip-duration",
        features=OrderedDict([
            ("vendor_id", FeatureType.CATEGORICAL),
            ("pickup_datetime", FeatureType.DATE),
            ("passenger_count", FeatureType.CONTINUOUS),
            ("pickup_longitude", FeatureType.CONTINUOUS),
            ("pickup_latitude", FeatureType.CONTINUOUS),
            ("dropoff_longitude", FeatureType.CONTINUOUS),
            ("dropoff_latitude", FeatureType.CONTINUOUS),
            ("store_and_fwd_flag", FeatureType.CATEGORICAL)]),
        target={"trip_duration": TargetType.REGRESSION},
        training_data_fname="train.zip",
        test_data_fname="test.zip",
        custom_preprocessor=None,
        scorer=scorers.root_mean_squared_log_error())


def mercedes_benz_greener_manufacturing():
    """Create data interface to kaggle 'mercedes benz greener manufacturing'.

    url: https://www.kaggle.com/c/mercedes-benz-greener-manufacturing
    """
    # these column numbers are not in the datasset (it's unclear why)
    exclude_nums = [7, 25, 72, 121, 149, 188, 193, 303, 381]
    return KaggleCompetition(
        competition_id="mercedes-benz-greener-manufacturing",
        features=OrderedDict(
            # NOTE: column "X9" does not exist.
            [("X%i" % i, FeatureType.CATEGORICAL)
             for i in range(9) if i not in exclude_nums] +
            [("X%i" % i, FeatureType.CONTINUOUS)
             for i in range(10, 386) if i not in exclude_nums]),
        target={"y": TargetType.REGRESSION},
        training_data_fname="train.csv.zip",
        test_data_fname="test.csv.zip",
        custom_preprocessor=None,
        scorer=scorers.r2_score())


def allstate_claims_severity():
    """Create data interface to kaggle 'allstate claims severity'.

    url: https://www.kaggle.com/c/allstate-claims-severity
    """
    return KaggleCompetition(
        competition_id="allstate-claims-severity",
        features=OrderedDict(
            # NOTE: column "X9" does not exist.
            [("cat%i" % (i + 1), FeatureType.CATEGORICAL)
             for i in range(116)] +
            [("cont%i" % (i + 1), FeatureType.CONTINUOUS)
             for i in range(14)]),
        target={"loss": TargetType.REGRESSION},
        training_data_fname="train.csv.zip",
        test_data_fname="test.csv.zip",
        custom_preprocessor=None,
        scorer=scorers.mean_absolute_error())


def house_prices_advanced_regression_techniques():
    """Create data interface to 'house prices advanced regression techniques'.

    url: https://www.kaggle.com/c/house-prices-advanced-regression-techniques
    """
    categorical_features = [
        "MSSubClass",
        "MSZoning",
        "Street",
        "Alley",
        "LotShape",
        "LandContour",
        "Utilities",
        "LotConfig",
        "LandSlope",
        "Neighborhood",
        "Condition1",
        "Condition2",
        "BldgType",
        "HouseStyle",
        "OverallQual",
        "OverallCond",
        "RoofStyle",
        "RoofMatl",
        "Exterior1st",
        "Exterior2nd",
        "MasVnrType",
        "ExterQual",
        "ExterCond",
        "Foundation",
        "BsmtQual",
        "BsmtCond",
        "BsmtExposure",
        "BsmtFinType1",
        "BsmtFinType2",
        "Heating",
        "HeatingQC",
        "CentralAir",
        "Electrical",
        "KitchenQual",
        "Functional",
        "FireplaceQu",
        "GarageType",
        "GarageFinish",
        "GarageQual",
        "GarageCond",
        "PavedDrive",
        "PoolQC",
        "Fence",
        "MiscFeature",
        "SaleType",
        "SaleCondition",
    ]
    continuous_features = [
        "LotFrontage",
        "LotArea",
        "YearBuilt",
        "MasVnrArea",
        "BsmtFinSF1",
        "BsmtFinSF2",
        "BsmtUnfSF",
        "TotalBsmtSF",
        "1stFlrSF",
        "2ndFlrSF",
        "LowQualFinSF",
        "GrLivArea",
        "BsmtFullBath",
        "BsmtHalfBath",
        "FullBath",
        "HalfBath",
        "TotRmsAbvGrd",
        "Fireplaces",
        "GarageYrBlt",
        "GarageCars",
        "GarageArea",
        "WoodDeckSF",
        "EnclosedPorch",
        "3SsnPorch",
        "ScreenPorch",
        "PoolArea",
        "MiscVal",
        "YrSold",
    ]
    return KaggleCompetition(
        competition_id="house-prices-advanced-regression-techniques",
        features=OrderedDict(
            [(c, FeatureType.CATEGORICAL) for c in categorical_features] +
            [(c, FeatureType.CONTINUOUS) for c in continuous_features]),
        target={"SalePrice": TargetType.REGRESSION},
        training_data_fname="train.csv.gz",
        test_data_fname="test.csv.gz",
        custom_preprocessor=None,
        scorer=scorers.root_mean_squared_log_error())
