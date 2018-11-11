"""Dash App to Visualize Deep-cash Experiments."""

import dash
import dash_core_components as dcc
import dash_html_components as html

import joblib
import os
import pandas as pd
import re

from dash.dependencies import Input, Output
from pathlib import Path

import plotting_helpers


OUTPUT_ROOT = Path(os.path.dirname(__file__)) / ".." / "floyd_outputs"

app = dash.Dash(__name__)
app.title = "Experiment Viewer"


def get_all_jobs():
    jobs = map(lambda x: int(x.name), OUTPUT_ROOT.glob("*"))
    jobs = reversed(sorted(jobs))
    return list(map(lambda x: {"label": str(x), "value": x}, jobs))


def generate_table(dataframe, max_rows=10):
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns])] +

        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ]) for i in range(min(len(dataframe), max_rows))]
    )


def read_results(job_nums, output_root=OUTPUT_ROOT):
    results = pd.concat([
        pd.read_csv(
            output_root /
            str(job_num) /
            "rnn_cash_controller_experiment.csv")
        .assign(job_number=job_num)
        .assign(job_trial_id=lambda df: df.job_number.astype(str).str.cat(
                df.trial_number.astype(str), sep="-"))
        .sort_values("episode")
        for job_num in job_nums
    ])
    return results


def preprocess_results(results):
    preprocessed_results = (
        results
        .set_index(["episode", "job_number"])
        .groupby("job_number")
        .apply(lambda df: df[plotting_helpers.METRICS].ewm(alpha=0.05).mean())
        .reset_index()
    )
    return preprocessed_results


def read_best_mlfs(job_nums, output_root=OUTPUT_ROOT):
    best_mlfs = []
    for job_num in job_nums:
        job_output_fp = OUTPUT_ROOT / str(job_num)
        for fp in job_output_fp.glob("cash_controller_mlfs_trial_*/*.pkl"):
            mlf = joblib.load(fp)
            episode = int(
                re.match("best_mlf_episode_(\d+).pkl", fp.name).group(1))
            mlf_str = "NONE" if mlf is None \
                else " > ".join(s[0] for s in mlf.steps)
            best_mlfs.append([job_num, episode, mlf_str])
    return pd.DataFrame(
        best_mlfs, columns=["job_number", "episode", "mlf"])


def _parse_job_choice(job_choice):
    if isinstance(job_choice, str):
        job_choice = [job_choice]
    if len(job_choice) == 0:
        return ""
    return list(map(int, job_choice))


app.layout = html.Div(children=[
    html.H1(children="Experiment Viewer"),
    html.Div(children="Analyze Deep Cash Experiments"),

    dcc.Dropdown(
        id="job-choices",
        options=get_all_jobs(),
        multi=True,
        value="219"),

    html.H2(children="Run History"),
    dcc.Graph(id="graph-run-history"),

    html.H2(children="Run History by Data Environment"),
    dcc.Dropdown(
        id="performance-metric",
        options=[
            {"label": m.replace("_", " "), "value": m}
            for m in plotting_helpers.METRICS
        ],
        value="mean_rewards"),
    dcc.Graph(id="graph-run-history-by-dataenv"),

    html.H2(children="Best MLFs per Episode"),
    dcc.Graph(id="best-mlfs"),

    html.Div(id="data-store", style={"display": "none"}),
])


@app.callback(
    Output("data-store", "children"), [Input("job-choices", "value")])
def preprocess_results_callback(job_choice):
    job_nums = _parse_job_choice(job_choice)
    results = read_results(job_nums)
    return results.to_json(date_format="iso", orient="split")


@app.callback(
    Output("graph-run-history", "figure"), [Input("data-store", "children")])
def plot_run_history_callback(data_store):
    if data_store == "":
        return {}
    return plotting_helpers.plot_run_history(
        pd.read_json(data_store, orient="split"))


@app.callback(
    Output("graph-run-history-by-dataenv", "figure"),
    [Input("data-store", "children"),
     Input("performance-metric", "value")])
def plot_run_history_by_dataenv_callback(data_store, performance_metric):
    if data_store == "":
        return {}
    return plotting_helpers.plot_run_history_by_dataenv(
        pd.read_json(data_store, orient="split"), metric=performance_metric)


@app.callback(
    Output("best-mlfs", "figure"), [Input("job-choices", "value")])
def plot_best_mlfs(job_choice):
    job_nums = _parse_job_choice(job_choice)
    best_mlfs = read_best_mlfs(job_nums)
    return plotting_helpers.plot_best_mlfs(best_mlfs)


if __name__ == "__main__":
    app.run_server(debug=True)
