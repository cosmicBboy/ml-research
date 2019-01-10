"""End to end tests for fitting a cash controllers."""

import numpy as np
import pandas as pd
import torch

from torch.autograd import Variable

from deep_cash import components

from deep_cash.algorithm_space import AlgorithmSpace, \
    CLASSIFIER_MLF_SIGNATURE, REGRESSOR_MLF_SIGNATURE
from deep_cash.task_environment import TaskEnvironment
from deep_cash.cash_controller import CASHController
from deep_cash.cash_reinforce import CASHReinforce
from deep_cash import utils


def _task_environment(
        target_types=["BINARY", "MULTICLASS"],
        dataset_names=["sklearn.iris", "sklearn.breast_cancer"],
        env_sources=["SKLEARN"],
        enforce_limits=True,
        n_samples=100):
    return TaskEnvironment(
        env_sources=env_sources,
        target_types=target_types,
        random_state=100,
        enforce_limits=enforce_limits,
        per_framework_time_limit=10,
        per_framework_memory_limit=1000,
        dataset_names=dataset_names,
        error_reward=0,
        n_samples=n_samples)


def _algorithm_space(classifiers=None, regressors=None):
    return AlgorithmSpace(
        classifiers=None,
        regressors=None,
        with_end_token=False,
        hyperparam_with_start_token=False,
        hyperparam_with_none_token=False)


def _cash_controller(a_space, t_env):
    return CASHController(
        metafeature_size=t_env.metafeature_dim,
        input_size=5,
        hidden_size=5,
        output_size=5,
        a_space=a_space,
        dropout_rate=0.2,
        num_rnn_layers=3)


def _cash_reinforce(controller, task_env, **kwargs):
    return CASHReinforce(
        controller,
        task_env,
        beta=0.9,
        metrics_logger=None,
        **kwargs)


def _fit_kwargs():
    return {
        "optim": torch.optim.Adam,
        "optim_kwargs": {"lr": 0.005},
        "n_iter": 3,
        "verbose": False
    }


def test_cash_reinforce_fit():
    """Ensure DeepCASH training routine executes."""
    n_episodes = 20
    t_env = _task_environment()
    a_space = _algorithm_space()
    controller = _cash_controller(a_space, t_env)
    reinforce = _cash_reinforce(controller, t_env, with_baseline=True)
    reinforce.fit(
        n_episodes=n_episodes,
        **_fit_kwargs())

    history = pd.DataFrame(reinforce.history())
    assert history.shape[0] == n_episodes
    for metric in [
            "mean_rewards",
            "mean_validation_scores",
            "best_validation_scores"]:
        assert (history[metric].dropna() <= 1).all()
        assert (history[metric].dropna() >= 0).all()
    for col in [
            "episode",
            "data_env_names",
            "scorers",
            "losses",
            "aggregate_gradients",
            "std_validation_scores",
            "n_successful_mlfs",
            "n_unique_mlfs",
            "n_unique_hyperparams",
            "mlf_diversity",
            "hyperparam_diversity",
            "best_mlfs"]:
        assert col in history


def test_cash_reinforce_fit_multi_baseline():
    """Make sure that baseline function maintains buffers per data env."""
    n_episodes = 5
    t_env = _task_environment()
    a_space = _algorithm_space()
    controller = _cash_controller(a_space, t_env)
    reinforce = _cash_reinforce(
        controller, t_env, with_baseline=True, single_baseline=False)
    reinforce.fit(
        n_episodes=n_episodes,
        **_fit_kwargs())
    assert reinforce._baseline_buffer_history["sklearn.iris"] != \
        reinforce._baseline_buffer_history["sklearn.wine"]


def test_cash_zero_gradient():
    """Test that gradient is zero if the reward is zero."""
    torch.manual_seed(100)  # ensure weight initialized is deterministic
    n_episodes = 10
    t_env = _task_environment()
    a_space = _algorithm_space()
    controller = _cash_controller(a_space, t_env)
    # don't train with baseline since this will modify the reward signal when
    # computing the `advantage = reward - baseline`.
    reinforce = _cash_reinforce(controller, t_env, with_baseline=False)
    fit_kwargs = _fit_kwargs()
    fit_kwargs.update({"n_iter": 1})
    reinforce.fit(
        n_episodes=n_episodes,
        **_fit_kwargs())
    # make sure there's at least one zero-valued aggregate gradient
    assert any([g == 0 for g in reinforce.aggregate_gradients])
    for r, g in zip(reinforce.mean_rewards, reinforce.aggregate_gradients):
        if r == 0:
            assert g == 0
        else:
            assert g != 0


def test_cash_entropy_regularizer():
    """Test that losses w/ entropy regularization are lower than baseline."""
    losses = {}
    for model, kwargs in [
            ("baseline", {
                "with_baseline": True,
                "entropy_coef": 0.0}),
            ("entropy_regularized", {
                "with_baseline": True,
                "entropy_coef": 0.5})]:
        torch.manual_seed(200)  # ensure weight initialized is deterministic
        # only run for a few episodes because model losses
        # become incomparable as models diverge
        n_episodes = 3
        t_env = _task_environment()
        a_space = _algorithm_space()
        controller = _cash_controller(a_space, t_env)
        reinforce = _cash_reinforce(controller, t_env, **kwargs)
        fit_kwargs = _fit_kwargs()
        fit_kwargs.update({"n_iter": 4})
        reinforce.fit(
            n_episodes=n_episodes,
            **_fit_kwargs())
        losses[model] = reinforce.losses
    assert (
        np.array(losses["entropy_regularized"]) <
        np.array(losses["baseline"])).all()


def test_cash_reinforce_regressor():
    """Test cash reinforce regression data environments."""
    n_episodes = 4
    for dataset in ["sklearn.boston", "sklearn.diabetes", "sklearn.linnerud"]:
        a_space = _algorithm_space()
        t_env = _task_environment(
            target_types=["REGRESSION"],
            dataset_names=[dataset])
        a_space = _algorithm_space()
        controller = _cash_controller(a_space, t_env)
        reinforce = _cash_reinforce(controller, t_env, with_baseline=True)
        reinforce.fit(
            n_episodes=n_episodes,
            **_fit_kwargs())
        history = pd.DataFrame(reinforce.history())
        assert history.shape[0] == n_episodes


def test_cash_missing_data():
    """Test cash reinforce on datasets with missing data."""
    a_space = _algorithm_space()
    X = np.array([
        [1, 5.1, 1],
        [2, np.nan, 1],
        [1, 6.1, 0],
        [5, np.nan, 0],
        [6, 1.1, 1],
        [6, 1.1, 1],
    ])
    for mlf_sig in [CLASSIFIER_MLF_SIGNATURE, REGRESSOR_MLF_SIGNATURE]:
        for i in range(20):
            mlf = a_space.sample_ml_framework(mlf_sig)
            imputer = mlf.named_steps["NumericImputer"]
            X_impute = imputer.fit_transform(X)
            assert (~np.isnan(X_impute)).all()


def test_cash_kaggle_regression_data():
    """Test regression dataset from kaggle."""
    n_episodes = 2
    a_space = _algorithm_space()
    t_env = _task_environment(
        env_sources=["KAGGLE"],
        target_types=["REGRESSION"],
        dataset_names=[
            "kaggle.restaurant_revenue_prediction",
            "kaggle.nyc_taxi_trip_duration",
            "kaggle.mercedes_benz_greener_manufacturing",
            "kaggle.allstate_claims_severity",
            "kaggle.house_prices_advanced_regression_techniques",
        ],
        n_samples=30)
    controller = _cash_controller(a_space, t_env)
    reinforce = _cash_reinforce(controller, t_env, with_baseline=True)
    reinforce.fit(
        n_episodes=n_episodes,
        **_fit_kwargs())
    history = pd.DataFrame(reinforce.history())
    assert history.shape[0] == n_episodes


def test_cash_kaggle_classification_data():
    """Test classification dataset from kaggle."""
    n_episodes = 2
    a_space = _algorithm_space()
    t_env = _task_environment(
        env_sources=["KAGGLE"],
        target_types=["BINARY", "MULTICLASS"],
        dataset_names=[
            "kaggle.homesite_quote_conversion",
            "kaggle.santander_customer_satisfaction",
            "kaggle.bnp_paribas_cardif_claims_management",
            "kaggle.poker_rule_induction",
            "kaggle.costa_rican_household_poverty_prediction",
        ],
        n_samples=30)
    controller = _cash_controller(a_space, t_env)
    reinforce = _cash_reinforce(controller, t_env, with_baseline=True)
    reinforce.fit(
        n_episodes=n_episodes,
        **_fit_kwargs())
    history = pd.DataFrame(reinforce.history())
    assert history.shape[0] == n_episodes


def test_cash_exclusion_mask():
    n_episodes = 10
    a_space = AlgorithmSpace(
        data_preprocessors=[
            components.data_preprocessors.impute_numeric(),
            components.data_preprocessors.one_hot_encoder(),
            components.data_preprocessors.standard_scaler(),
        ],
        feature_preprocessors=[
            components.feature_preprocessors.pca(),
        ],
        classifiers=[
            components.classifiers.logistic_regression(),
            components.classifiers.support_vector_classifier_linear(),
        ])
    t_env = _task_environment(
        env_sources=["SKLEARN"],
        target_types=["BINARY"],
        dataset_names=["sklearn.breast_cancer"],
        n_samples=20)

    controller = _cash_controller(a_space, t_env)

    t_env.sample_data_env()
    prev_reward, prev_action = 0, controller.init_action()
    for i in range(200):
        metafeature_tensor = t_env.sample_task_state()
        target_type = t_env.current_data_env.target_type
        actions, action_activation = controller.decode(
            init_input_tensor=prev_action,
            target_type=target_type,
            aux=utils.aux_tensor(prev_reward),
            metafeatures=Variable(metafeature_tensor),
            init_hidden=controller.init_hidden())
        for action in actions:
            if action["action_name"] in controller._exclude_masks:
                exclude_mask = controller._exclude_masks[
                    action["action_name"]].tolist()
                assert exclude_mask[action["choice_index"]] == 0
