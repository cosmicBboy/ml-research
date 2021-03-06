"""End to end tests for fitting a cash controllers."""

import numpy as np
import pandas as pd
import pytest
import torch

from metalearn import components

from metalearn.algorithm_space import AlgorithmSpace, \
    CLASSIFIER_MLF_SIGNATURE, REGRESSOR_MLF_SIGNATURE
from metalearn.data_environments import openml_api
from metalearn.task_environment import TaskEnvironment
from metalearn.metalearn_controller import MetaLearnController
from metalearn.metalearn_reinforce import MetaLearnReinforce
from metalearn.random_search import CASHRandomSearch
from metalearn import utils


def _task_environment(
        target_types=[
            "BINARY",
            "MULTICLASS"
        ],
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
        classifiers=classifiers,
        regressors=regressors,
        data_preprocessors=[
            components.data_preprocessors.simple_imputer(),
            components.data_preprocessors.one_hot_encoder(),
            components.data_preprocessors.standard_scaler(),
        ],
        with_end_token=False,
        hyperparam_with_start_token=False,
        hyperparam_with_none_token=False,
        random_state=2001)


def _metalearn_controller(a_space, t_env):
    return MetaLearnController(
        metafeature_size=t_env.metafeature_dim,
        input_size=5,
        hidden_size=5,
        output_size=5,
        a_space=a_space,
        dropout_rate=0.2,
        num_rnn_layers=1)


def _metalearn_reinforce(controller, task_env, **kwargs):
    return MetaLearnReinforce(
        controller,
        task_env,
        gamma=0.9,
        metrics_logger=None,
        **kwargs)


def _fit_kwargs():
    return {
        "optim": torch.optim.Adam,
        "optim_kwargs": {"lr": 0.005, "weight_decay": 0.01},
        "n_iter": 3,
        "verbose": False
    }


def test_metalearn_reinforce_fit():
    """Ensure MetaLearn training routine executes."""
    n_episodes = 20
    t_env = _task_environment()
    a_space = _algorithm_space()
    controller = _metalearn_controller(a_space, t_env)
    reinforce = _metalearn_reinforce(controller, t_env)
    reinforce.fit(
        n_episodes=n_episodes,
        **_fit_kwargs())

    history = pd.DataFrame(reinforce.history)
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
            "total_losses",
            "actor_losses",
            "critic_losses",
            "entropy_losses",
            "gradient_norms",
            "std_validation_scores",
            "n_successful_mlfs",
            "n_unique_mlfs",
            "n_unique_hyperparams",
            "mlf_diversity",
            "hyperparam_diversity"]:
        assert col in history


def test_metalearn_entropy_regularizer():
    """Test that losses w/ entropy regularization are lower than baseline."""
    losses = {}
    for model, kwargs in [
            ("baseline", {"entropy_coef": 0.0}),
            ("entropy_regularized", {"entropy_coef": 0.5})]:
        torch.manual_seed(200)  # ensure weight initialized is deterministic
        # only run for a few episodes because model losses
        # become incomparable as models diverge
        n_episodes = 3
        t_env = _task_environment()
        a_space = _algorithm_space()
        controller = _metalearn_controller(a_space, t_env)
        reinforce = _metalearn_reinforce(controller, t_env, **kwargs)
        fit_kwargs = _fit_kwargs()
        fit_kwargs.update({"n_iter": 4})
        reinforce.fit(
            n_episodes=n_episodes,
            **_fit_kwargs())
        losses[model] = reinforce.tracker.history["total_losses"]
    assert (
        np.array(losses["entropy_regularized"]) <
        np.array(losses["baseline"])).all()


def test_metalearn_reinforce_regressor():
    """Test cash reinforce regression data environments."""
    n_episodes = 10
    for dataset in ["sklearn.boston", "sklearn.diabetes", "sklearn.linnerud"]:
        a_space = _algorithm_space()
        t_env = _task_environment(
            target_types=["REGRESSION", "MULTIREGRESSION"],
            dataset_names=[dataset])
        controller = _metalearn_controller(a_space, t_env)
        reinforce = _metalearn_reinforce(controller, t_env)
        reinforce.fit(
            n_episodes=n_episodes,
            **_fit_kwargs())
        history = pd.DataFrame(reinforce.history)
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
            mlf = a_space.sample_ml_framework(
                mlf_sig, task_metadata={"continuous_features": [0, 1, 2]})
            imputer = mlf.named_steps["SimpleImputer"]
            X_impute = imputer.fit_transform(X)
            assert (~np.isnan(X_impute)).all()


@pytest.mark.skip(reason="run-time too long")
def test_kaggle_regression_data():
    """Test regression dataset from kaggle."""
    n_episodes = 20
    a_space = _algorithm_space()
    for dataset_name in [
        "kaggle.restaurant_revenue_prediction",
        "kaggle.nyc_taxi_trip_duration",
        "kaggle.mercedes_benz_greener_manufacturing",
        "kaggle.allstate_claims_severity",
        "kaggle.house_prices_advanced_regression_techniques",
    ]:
        t_env = _task_environment(
            env_sources=["KAGGLE"],
            target_types=["REGRESSION"],
            dataset_names=[dataset_name],
            n_samples=500)
        controller = _metalearn_controller(a_space, t_env)
        reinforce = _metalearn_reinforce(controller, t_env)
        reinforce.fit(
            n_episodes=n_episodes,
            **_fit_kwargs())
        history = pd.DataFrame(reinforce.history)
        assert history.shape[0] == n_episodes
        assert history["n_successful_mlfs"].sum() > 0


@pytest.mark.skip(reason="run-time too long")
def test_kaggle_classification_data():
    """Test classification dataset from kaggle."""
    torch.manual_seed(100)
    n_episodes = 20
    a_space = _algorithm_space()
    for dataset_name in [
        "kaggle.homesite_quote_conversion",
        "kaggle.santander_customer_satisfaction",
        "kaggle.bnp_paribas_cardif_claims_management",
        "kaggle.poker_rule_induction",
        "kaggle.costa_rican_household_poverty_prediction",
    ]:
        t_env = _task_environment(
            env_sources=["KAGGLE"],
            target_types=["BINARY", "MULTICLASS"],
            dataset_names=[dataset_name],
            n_samples=500)
        controller = _metalearn_controller(a_space, t_env)
        reinforce = _metalearn_reinforce(controller, t_env)
        reinforce.fit(
            n_episodes=n_episodes,
            **_fit_kwargs())
        history = pd.DataFrame(reinforce.history)
        assert history.shape[0] == n_episodes
        assert history["n_successful_mlfs"].sum() > 0


@pytest.mark.skip(reason="run-time too long")
def test_openml_regression_data():
    n_episodes = 20
    a_space = _algorithm_space()
    datasets = openml_api.regression_envs(
        n=openml_api.N_REGRESSION_ENVS)
    for dataset_name in datasets.keys():
        t_env = _task_environment(
            env_sources=["OPEN_ML"],
            target_types=["REGRESSION"],
            dataset_names=[dataset_name],
            n_samples=100)
        controller = _metalearn_controller(a_space, t_env)
        reinforce = _metalearn_reinforce(controller, t_env)
        reinforce.fit(
            n_episodes=n_episodes,
            **_fit_kwargs())
        history = pd.DataFrame(reinforce.history)
        assert history.shape[0] == n_episodes
        assert history["n_successful_mlfs"].sum() > 0


@pytest.mark.skip(reason="run-time too long")
def test_openml_classification_data():
    n_episodes = 20
    a_space = _algorithm_space()
    datasets = openml_api.classification_envs(
        n=openml_api.N_CLASSIFICATION_ENVS)
    for dataset_name in datasets.keys():
        t_env = _task_environment(
            env_sources=["OPEN_ML"],
            target_types=["BINARY", "MULTICLASS"],
            dataset_names=[dataset_name],
            n_samples=1000)
        controller = _metalearn_controller(a_space, t_env)
        reinforce = _metalearn_reinforce(controller, t_env)
        reinforce.fit(
            n_episodes=n_episodes,
            **_fit_kwargs())
        history = pd.DataFrame(reinforce.history)
        assert history.shape[0] == n_episodes
        assert history["n_successful_mlfs"].sum() > 0


def _exclusion_mask_test_harness(n_episodes, a_space_kwargs, t_env_kwargs):
    a_space = AlgorithmSpace(
        data_preprocessors=[
            components.data_preprocessors.simple_imputer(),
            components.data_preprocessors.one_hot_encoder(),
            components.data_preprocessors.standard_scaler(),
        ],
        **a_space_kwargs)
    t_env = _task_environment(
        env_sources=["SKLEARN"], n_samples=20, **t_env_kwargs)
    controller = _metalearn_controller(a_space, t_env)
    prev_reward, prev_action = 0, controller.init_action()
    # NOTE: this is a pared down version of how the reinforce fitter implements
    # the fit method
    t_env.sample_data_env()
    prev_hidden = controller.init_hidden()
    for i in range(n_episodes):
        metafeature_tensor = t_env.sample_task_state()
        target_type = t_env.current_data_env.target_type
        value, actions, action_activation, hidden = controller(
            prev_action=prev_action,
            prev_reward=utils.aux_tensor(prev_reward),
            metafeatures=metafeature_tensor,
            hidden=prev_hidden,
            target_type=target_type,
        )

        prev_hidden = hidden
        for action in actions:
            # exclude masks are 1 to ignore the action and 0 to include in
            # in the candidates of actions to sample from.
            if action["action_name"] in controller._exclude_masks:
                exclude_mask = controller._exclude_masks[
                    action["action_name"]].tolist()
                try:
                    assert any(i == 0 for i in exclude_mask)
                except:
                    import ipdb; ipdb.set_trace()
                # make sure that the chosen action is not masked
                assert exclude_mask[action["choice_index"]] == 0


def test_cash_classifier_exclusion_masks():
    """Test classifier exclusion mask logic."""
    _exclusion_mask_test_harness(
        100,
        a_space_kwargs={
            "feature_preprocessors": [components.feature_preprocessors.pca()],
            "classifiers": [
                components.classifiers.logistic_regression(),
                components.classifiers.support_vector_classifier_linear()
            ]
        },
        t_env_kwargs={
            "target_types": ["BINARY"],
            "dataset_names": ["sklearn.breast_cancer"],
        })


def test_cash_regressor_exclusion_masks():
    """Test regression exclusion mask logic."""
    _exclusion_mask_test_harness(
        100,
        a_space_kwargs={
            "feature_preprocessors": [components.feature_preprocessors.pca()],
            "regressors": [
                components.regressors.support_vector_regression_nonlinear()
            ]
        },
        t_env_kwargs={
            "target_types": ["REGRESSION"],
            "dataset_names": ["sklearn.diabetes"],
        })


def test_cash_feature_processor_exclusion_masks():
    """Test feature processor exclusion mask logic."""
    _exclusion_mask_test_harness(
        100,
        a_space_kwargs={
            "feature_preprocessors": [
                components.feature_preprocessors.kernel_pca(),
                components.feature_preprocessors.nystroem_sampler()
            ],
            "classifiers": [components.classifiers.k_nearest_neighbors()],
            "regressors": [
                components.regressors.k_nearest_neighbors_regression()
            ]
        },
        t_env_kwargs={
            "target_types": ["BINARY", "REGRESSION"],
            "dataset_names": ["sklearn.breast_cancer", "sklearn.diabetes"],
        })


def test_random_search():
    """Test random search CASH object."""
    cash_random = CASHRandomSearch(
        _algorithm_space(),
        _task_environment())
    cash_random.fit(n_episodes=5, n_iter=5)
    assert all([len(x) == 5 for x in cash_random.history.values()])
