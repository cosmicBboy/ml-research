"""Dash App to Visualize Deep-cash Experiments."""

import dash
import dash_core_components as dcc
import dash_html_components as html

import os
import pandas as pd

from dash.dependencies import Input, Output
from pathlib import Path

import plotting_helpers

app = dash.Dash(__name__)


def generate_table(dataframe, max_rows=10):
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns])] +

        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ]) for i in range(min(len(dataframe), max_rows))]
    )


def read_results(job_nums, output_root=Path(os.path.dirname(__file__))):
    results = pd.concat([
        pd.read_csv(
            output_root /
            ".." /
            "floyd_outputs" /
            str(job_num) /
            "rnn_cash_controller_experiment.csv")
        .assign(job_number=job_num)
        .assign(job_trial_id=lambda df: df.job_number.astype(str).str.cat(
                df.trial_number.astype(str), sep="-"))
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


def graph_data():
    return {
        "data": [
            {"x": [1, 2, 3], "y": [4, 1, 2], "type": "bar", "name": "SF"},
            {"x": [1, 2, 3], "y": [2, 4, 5], "type": "bar", "name": "NY"},
        ],
        "layout": {
            "title": "Graph 1"
        }
    }


global_results = read_results([212])


app.layout = html.Div(children=[
    html.H1(children="Experiment Viewer"),
    html.Div(children="Analyze Deep Cash Experiments"),

    dcc.Checklist(
        id="job-choices",
        options=[
            {"label": "Job 212", "value": "212"}
        ],
        values=["212"]),

    html.H2(children="Run History"),
    dcc.Graph(id="graph-run-history"),

    html.H2(children="Run History by Data Environment"),
    dcc.Graph(id="graph-run-history-by-dataenv"),

    html.Div(id="data-store", style={"display": "none"}),
])


@app.callback(
    Output("data-store", "children"), [Input("job-choices", "values")])
def preprocess_results_callback(values):
    if len(values) == 0:
        return {}
    values = list(map(int, values))
    results = global_results[
        global_results.job_number.isin(values)]
    return results.to_json(date_format="iso", orient="split")


@app.callback(
    Output("graph-run-history", "figure"), [Input("data-store", "children")])
def plot_run_history_callback(data_store):
    return plotting_helpers.plot_run_history(
        pd.read_json(data_store, orient="split"))


@app.callback(
    Output("graph-run-history-by-dataenv", "figure"),
    [Input("data-store", "children")])
def plot_run_history_by_dataenv_callback(data_store):
    return plotting_helpers.plot_run_history_by_dataenv(
        pd.read_json(data_store, orient="split"))


if __name__ == "__main__":
    app.run_server(debug=True)
