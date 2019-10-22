"""View experiments by job."""

import streamlit as st
import numpy as np
import pandas as pd
import subprocess

from floyd.client.experiment import ExperimentClient
from floyd.client.data import DataClient
from pathlib import Path
from torch import tensor

from metalearn import plotting


experiment_client = ExperimentClient()
data_client = DataClient()

cache_dir = Path.home() / "floyd_cache"

EXPERIMENT_LIMIT = 10000
SUCCESS_STATE = "success"
METRICS_FILE = "rnn_metalearn_controller_experiment.csv"


@st.cache
def get_experiments():
    return {
        exp.name: exp for exp in
        experiment_client.get_all(limit=EXPERIMENT_LIMIT)
        if exp.state == SUCCESS_STATE
    }


@st.cache
def get_experiment_data(experiment_name: str) -> Path:
    path = cache_dir / experiment_name
    path.mkdir(exist_ok=True, parents=True)
    subprocess.check_call(
        ["floyd", "data", "clone", f"{experiment_name}/output"],
        cwd=path)
    return path


@st.cache
def read_results_file(experiment_path: Path) -> pd.DataFrame:
    return pd.read_csv(experiment_path / METRICS_FILE)


def sidebar():
    pass


def main():
    """Run the app."""

    """
    # MetaLearn Experiment Viewer

    Select experiments
    """

    experiments = get_experiments()
    experiment_names = st.multiselect(
        label='Select experiments to compare',
        options=sorted(
            experiments,
            key=lambda x: int(x.split("/")[-1]),
            reverse=True,
        ),
    )
    experiment_data_paths = [
        get_experiment_data(name) for name in experiment_names]
    if len(experiment_data_paths) > 0:

        results = pd.concat([
            read_results_file(path).assign(job_number=path.name)
            for path in experiment_data_paths
        ]).assign(aggregate_gradients=lambda df: df.aggregate_gradients.map(
            lambda x: eval(x)))
        reward_by_data_env = results.pivot(
            index="episode", columns="data_env_names", values="mean_rewards")
        reward_by_data_env.index.name = "index"

        st.write(f"Results data")
        st.write(results)

        st.plotly_chart(plotting.plot_run_history(results))

        st.markdown("""
        <br><br>
        """, unsafe_allow_html=True)

        """
        ### Metrics by task
        """

        metric = st.selectbox(
            label="Select metric to plot",
            options=[
                col for col in results if col not in [
                    "data_env_names", "episode", "scorers"]
            ]
        )

        st.plotly_chart(
            plotting.plot_run_history_by_dataenv(
                results, metric=metric))


if __name__ == "__main__":
    main()
