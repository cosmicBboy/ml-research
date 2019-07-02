"""Metadata for kaggle classification datasets."""

import itertools

from collections import OrderedDict

from .kaggle_base import KaggleCompetition

from ..data_types import FeatureType, TargetType
from .. import scorers

from .feature_maps import (
    kaggle_costa_rican_household_poverty_prediction,
    kaggle_homesite_quote_conversion,
    kaggle_santander_customer_satisfaction,
)


def homesite_quote_conversion():
    """Create data interface to kaggle 'homesite quote conversion'.

    url: https://www.kaggle.com/c/homesite-quote-conversion
    """
    features = kaggle_homesite_quote_conversion.get_feature_map()

    def _preprocessor(df):
        """Deal with quirk in the dataset where floats have commas."""
        for f, ftype in features.items():
            if ftype is FeatureType.CONTINUOUS:
                if df[f].dtype == "object":
                    x = df[f].str.replace(",", "")
                    df[f] = x.astype(float)
        return df

    return KaggleCompetition(
        competition_id="homesite-quote-conversion",
        features=features,
        target={"QuoteConversion_Flag": TargetType.BINARY},
        training_data_fname="train.csv.zip",
        test_data_fname="test.csv.zip",
        custom_preprocessor=_preprocessor,
        scorer=scorers.roc_auc())


def santander_customer_satisfaction():
    """Create data interface to kaggle 'santander customer satisfaction'.

    url: https://www.kaggle.com/c/santander-customer-satisfaction
    """
    return KaggleCompetition(
        competition_id="santander-customer-satisfaction",
        features=kaggle_santander_customer_satisfaction.get_feature_map(),
        target={"TARGET": TargetType.BINARY},
        training_data_fname="train.csv.zip",
        test_data_fname="test.csv.zip",
        custom_preprocessor=None,
        scorer=scorers.roc_auc())


def bnp_paribas_cardif_claims_management():
    """Create data interface to kaggle 'bnp paribas cardif claims management'.

    url: https://www.kaggle.com/c/bnp-paribas-cardif-claims-management
    """
    categorical_features = [
        "v%d" % i for i in itertools.chain(
            [3, 22, 24],
            [30, 31, 47, 52, 56, 66, 71, 74, 75, 79, 91, 107, 110, 112, 113,
             125])]
    continuous_features = [
        "v%d" % i for i in itertools.chain(
            [1, 2],
            range(4, 22),
            [23],
            range(25, 30),
            range(32, 47),
            range(48, 52),
            range(53, 56),
            range(57, 66),
            range(67, 71),
            [72, 73, 76, 77, 78],
            range(80, 91),
            range(92, 107),
            [108, 109, 111],
            range(114, 125),
            range(126, 132))]

    return KaggleCompetition(
        competition_id="bnp-paribas-cardif-claims-management",
        features=OrderedDict(
            itertools.chain(
                ((f, FeatureType.CATEGORICAL) for f in categorical_features),
                ((f, FeatureType.CONTINUOUS) for f in continuous_features)
            )),
        target={"target": TargetType.BINARY},
        training_data_fname="train.csv.zip",
        test_data_fname="test.csv.zip",
        custom_preprocessor=None,
        scorer=scorers.log_loss())


def poker_rule_induction():
    """Create data interface to kaggle 'poker rule induction'.

    url: https://www.kaggle.com/c/poker-rule-induction
    """
    letters = [
        "S",  # suit: 1 - 13 (Ace, 2, 3, ..., King)
        "C",  # card: 1 - 4 (Hearts, Spades, Diamonds, Clubs)
    ]
    return KaggleCompetition(
        competition_id="poker-rule-induction",
        features=OrderedDict([
            ("%s%d" % (i, j), FeatureType.CONTINUOUS)
            for i in letters for j in range(1, 6)
        ]),
        target={"hand": TargetType.MULTICLASS},
        training_data_fname="train.csv.zip",
        test_data_fname="test.csv.zip",
        custom_preprocessor=None,
        scorer=scorers.accuracy())


def costa_rican_household_poverty_prediction():
    """Create data interface to kaggle 'costa rican household poverty prediction'.

    url: https://www.kaggle.com/c/costa-rican-household-poverty-prediction
    """
    def _preprocessor(df):
        """Deal with quirks in the costa rican household poverty dataset.

        https://www.kaggle.com/c/costa-rican-household-poverty-prediction/data

        This function deals with a quirk in the dataset where "yes" and "no"
        values are present in the following continuous columns:
        - dependency
        - edjefe
        - edjefa

        "yes" values are meant to be 1, and "no" values are meant to be 0
        """
        replace_dict = {"yes": "1", "no": "0"}
        return (
            df.replace({
                "dependency": replace_dict,
                "edjefe": replace_dict,
                "edjefa": replace_dict,
            })
            .astype({
                "dependency": float,
                "edjefe": float,
                "edjefa": float
            })
        )

    return KaggleCompetition(
        competition_id="costa-rican-household-poverty-prediction",
        features=kaggle_costa_rican_household_poverty_prediction.get_feature_map(),  # noqa E501
        target={"Target": TargetType.MULTICLASS},
        training_data_fname="train.csv.zip",
        test_data_fname="test.csv.zip",
        custom_preprocessor=_preprocessor,
        scorer=scorers.f1_score_macro())
