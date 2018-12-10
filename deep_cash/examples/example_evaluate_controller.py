"""Evaluate controller after training."""
import pandas as pd
import operator
import os
import torch

from collections import OrderedDict
from pathlib import Path
from sklearn.externals import joblib

from deep_cash.cash_controller import CASHController
from deep_cash.inference.inference_engine import CASHInference
from deep_cash.task_environment import TaskEnvironment
from deep_cash import utils, scorers


build_path = Path(os.path.dirname(__file__)) / ".." / "floyd_outputs" / "221"

controller = CASHController.load(build_path / "controller_trial_0.pt")
experiment_results = pd.read_csv(
    build_path / "rnn_cash_controller_experiment.csv")
base_mlf_path = build_path / "cash_controller_mlfs_trial_0"

# preprocess the results file, since there was a dataset naming convention
# change that needs to be taken account of.
experiment_results["data_env_names"] = experiment_results[
    "data_env_names"].map(lambda x: "kaggle.%s" % x)

# will need to make this part more elegant. But for now assume that we know
# each dataset's corresponding scorer. Ideally, which scorer was used for a
# particular episode would just be metadata in the
# rnn_cash_controller_experiment.csv file.
dataset_scorers = {
    "kaggle.bnp_paribas_cardif_claims_management": scorers.log_loss(),
    "kaggle.costa_rican_household_poverty_prediction":
        scorers.f1_score_macro(),
    "kaggle.homesite_quote_conversion": scorers.roc_auc(),
    "kaggle.poker_rule_induction": scorers.accuracy(),
    "kaggle.santander_customer_satisfaction": scorers.roc_auc(),
}

# just make simple assumption that if comparator is not operator.gt, it can
# only be operator.lt. This comparator is a function taking two args. Returns
# true if first arg is better than second arg.
experiment_results = experiment_results[
    experiment_results["data_env_names"].isin(dataset_scorers.keys())]
experiment_results["higher_score_is_better"] = (
    experiment_results.data_env_names.map(
        lambda x: dataset_scorers[x].comparator is operator.gt))

# get top 10 best mlfs for each data env across all episodes.
best_mlf_episodes = (
    experiment_results
    .groupby("data_env_names")
    .apply(lambda df: (
        df.sort_values(
            "best_validation_scores",
            ascending=df.higher_score_is_better.iloc[0] is False)
        .head(10))
    )
    ["episode"]
    .reset_index(level=1, drop=True)
)

# a dict mapping datasets to the top 10 mlfs found for those datasets.
best_mlfs = (
    best_mlf_episodes.map(
        lambda x: joblib.load(base_mlf_path / ("best_mlf_episode_%d.pkl" % x)))
    .groupby("data_env_names")
    .apply(lambda x: list(x))
    .to_dict()
)

# modify this controller so that it supports the NULL data env. This shouldn't
# be necessary in the future.
torch.manual_seed(10)
controller = utils.add_metafeatures_hidden_units(controller)

task_env = TaskEnvironment(
    env_sources=["KAGGLE", "SKLEARN"],
    test_set_config={"KAGGLE": {"test_size": 0.8, "random_state": 100}},
    random_state=100,
    enforce_limits=True,
    per_framework_time_limit=720,
    per_framework_memory_limit=10000,
    dataset_names=list(sorted(dataset_scorers.keys())),
    test_dataset_names=["sklearn.boston"],
    error_reward=0,
    target_types=["BINARY", "MULTICLASS"])

inference_engine = CASHInference(controller, task_env)

# evaluate controller on the test set of the training data environment
# distribution.
# TODO: this part still needs to ironed out. Right now CASHReinforce does not
# serialize the fitted models. Need to do that first in order to evaluate the
# test set data environments using models with best validation scores.
test_set_results = OrderedDict()
for dataset, mlfs in best_mlfs.items():
    test_set_results[dataset] = inference_engine.evaluate_test_sets(
        dataset, mlfs)

# evaluate controller on test data environments
train_env_results = inference_engine.evaluate_training_data_envs(
    n=1, datasets=["kaggle.restaurant_revenue_prediction"], verbose=True)
test_env_results = inference_engine.evaluate_test_data_envs(verbose=True)
