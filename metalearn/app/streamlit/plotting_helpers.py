"""Helper functions for plotly graphs."""

import colorlover
import math
import plotly.graph_objs as go

from collections import defaultdict
from plotly import tools

PALETTE = colorlover.scales["8"]["qual"]["Dark2"]
METRICS = [
    "losses",
    "aggregate_gradients",
    "best_validation_scores",
    "mean_rewards",
    "mean_validation_scores",
    "n_successful_mlfs",
    "mlf_diversity",
    "hyperparam_diversity",
]


def subplot_coords(iterable, ncols, one_indexed=True):
    n = len(iterable)
    nrows = math.ceil(n / ncols)
    offset = 1 if one_indexed else 0
    return {
        "nrows": nrows,
        "ncols": ncols,
        "coords":  [(i + offset, j + offset)
                    for i in range(nrows)
                    for j in range(ncols)]
    }


def create_time_series(x, y, group_name, showlegend, group_colormap=None):
    line_dict = dict(width=1)
    if group_colormap is not None:
        line_dict.update(dict(color=group_colormap[group_name]))
    return go.Scatter(
        x=x,
        y=y,
        name=group_name,
        legendgroup=group_name,
        mode='lines',
        line=line_dict,
        showlegend=showlegend,
        opacity=0.7,
    )


def create_multi_time_series(
        results, group_column, metric, legend_metric="mlf_diversity"):
    groups = results[group_column].unique()
    cm = {g: PALETTE[i] for i, g in enumerate(groups)}
    showlegend = True if metric == legend_metric else False
    return (
        results
        .groupby(group_column)
        .apply(lambda df: create_time_series(
            df["episode"],
            df[metric],
            df[group_column].iloc[0],
            showlegend,
            cm
        ))
        .tolist())


def plot_run_history(results):
    coords = subplot_coords(METRICS, 2)
    fig = tools.make_subplots(
        rows=coords["nrows"],
        cols=coords["ncols"],
        subplot_titles=METRICS,
        vertical_spacing=0.1,
        print_grid=False)

    for i, metric in enumerate(METRICS):
        traces = create_multi_time_series(
            results.astype({"job_number": str}),
            "job_number",
            metric,
        )
        row_i = coords["coords"][i][0]
        col_i = coords["coords"][i][1]
        for trace in traces:
            fig.append_trace(trace, row_i, col_i)
        # add x-axis titles on the bottom ncols plots
        if i >= (coords["ncols"] * coords["nrows"] - coords["ncols"]):
            xax = "xaxis%s" % ("" if i == 0 else i + 1)
            fig.layout[xax].update({"title": "episode"})

    fig.layout.update({
        "height": 600,
    })
    for annotation in fig.layout.annotations:
        annotation.font.update({"size": 12})
    return fig


def plot_run_history_by_dataenv(results, metric="mean_rewards", ncols=3):

    colormap = {
        g: PALETTE[i] for i, g in
        enumerate(results.job_number.unique())}

    def time_series(df, y, legend_metric="anneal"):
        line_dict = dict(width=1)
        job_number = df["job_number"].iloc[0]
        env_name = df["data_env_names"].iloc[0]
        color = colormap.get(job_number)
        showlegend = True if env_name == legend_metric else False
        if color is not None:
            line_dict.update(dict(color=color))
        return go.Scatter(
            x=df["episode"],
            y=df[y],
            name=str(job_number),
            legendgroup=str(job_number),
            line=line_dict,
            mode='lines',
            showlegend=showlegend,
        )

    # time_series_data is a dict where the key is
    # the env_name and value is the corresponding
    # trace.
    _time_series_data = (
        results
        .groupby(["data_env_names", "job_number"])
        .apply(time_series, y=metric)
        .to_dict()
    )

    time_series_data = defaultdict(dict)
    for (data_env, job_num), trace in _time_series_data.items():
        time_series_data[data_env][job_num] = trace

    coords = subplot_coords(time_series_data, ncols)
    fig = tools.make_subplots(
        rows=coords["nrows"],
        cols=coords["ncols"],
        subplot_titles=list(time_series_data.keys()),
        vertical_spacing=0.05,
        print_grid=False)

    for i, (data_env, traces) in enumerate(time_series_data.items()):
        row_i, col_i = coords["coords"][i][0], coords["coords"][i][1]
        for job_num, trace in traces.items():
            fig.append_trace(trace, row_i, col_i)
            # add x-axis titles on the bottom ncols plots
            if i >= (coords["ncols"] * coords["nrows"] - coords["ncols"]):
                xax = "xaxis%s" % ("" if i == 0 else i + 1)
                fig.layout[xax].update({"title": "episode"})

    fig.layout.update({
        "height": 1000,
    })

    for annotation in fig.layout.annotations:
        annotation.font.update({"size": 12})

    return fig


def plot_best_mlfs(best_mlfs):
    def create_best_mlf_timeline(x, y, color):
        return go.Scatter(
            x=x,
            y=y,
            mode='markers',
            opacity=0.7,
            line=dict(width=1, color=color)
        )

    colormap = {
        g: PALETTE[i] for i, g in
        enumerate(best_mlfs.job_number.unique())}

    traces = (
        best_mlfs.groupby("job_number")
        .apply(lambda df: create_best_mlf_timeline(
            df.episode, df.mlf, colormap.get(df.name)))
    ).tolist()

    fig = go.Figure(
        data=traces,
        layout=dict(
            height=500,
            margin=dict(l=600),
            hovermode="closest"
        ))

    fig.layout["xaxis"].update({"title": "episode"})
    return fig
