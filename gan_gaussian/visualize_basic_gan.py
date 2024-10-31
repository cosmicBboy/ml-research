"""Create plots of basic gan results."""

import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt

blues_cmap = ListedColormap(sns.color_palette("Blues_r", n_colors=20))
oranges_cmap = ListedColormap(sns.color_palette("Oranges_r", n_colors=20))

sns.set_style("white")


def create_grid_spec(figsize=(11, 11), wspace=0.5, hspace=2):
    fig = plt.figure(figsize=figsize)
    fig.suptitle(
        "GAN learns to generate a gaussian distribution",
        y=0.95,
        fontsize=20,
        fontweight="bold",
    )

    gs = GridSpec(14, 4)
    gs.update(wspace=wspace, hspace=hspace)
    ax1 = plt.subplot(gs[:6, :])
    ax2 = plt.subplot(gs[6:10, 0])
    ax3 = plt.subplot(gs[6:10, 1])
    ax4 = plt.subplot(gs[6:10, 2])
    ax5 = plt.subplot(gs[6:10, 3])
    ax6 = plt.subplot(gs[10:, :])
    return fig, gs, ax1, ax2, ax3, ax4, ax5, ax6


def get_sample_columms(run_samples):
    return [
        c
        for c in run_samples.columns
        if c.startswith("sample_") and c != "sample_type"
    ]


def plot_axes(
    real_samples,
    fake_samples,
    run_metadata,
    ax1,
    ax2,
    ax3,
    ax4,
    ax5,
    ax6,
    scatter_size=2.5,
):
    real_samples.plot.hist(ax=ax1, alpha=0.4)
    fake_samples.plot.hist(ax=ax1, alpha=0.4)
    ax1.set_xlabel("sample value")

    run_metadata = run_metadata.assign(epoch=lambda df: df.epoch / 1000)

    # plot real and fake mu
    run_metadata.plot(
        ax=ax2,
        x="epoch",
        y="real_mu",
        kind="scatter",
        colormap=blues_cmap,
        colorbar=False,
        c=run_metadata["epoch"],
        alpha=0.4,
        s=scatter_size,
    )
    run_metadata.plot(
        ax=ax2,
        x="epoch",
        y="fake_mu",
        kind="scatter",
        colormap=oranges_cmap,
        colorbar=False,
        c=run_metadata["epoch"],
        alpha=0.4,
        s=scatter_size,
    )

    # plot real and fake sigma
    run_metadata.plot(
        ax=ax3,
        x="epoch",
        y="real_sigma",
        kind="scatter",
        colormap=blues_cmap,
        colorbar=False,
        c=run_metadata["epoch"],
        alpha=0.5,
        s=scatter_size,
    )
    run_metadata.plot(
        ax=ax3,
        x="epoch",
        y="fake_sigma",
        kind="scatter",
        colormap=oranges_cmap,
        colorbar=False,
        c=run_metadata["epoch"],
        alpha=0.5,
        s=scatter_size,
    )

    # plot real and fake skew
    run_metadata.plot(
        ax=ax4,
        x="epoch",
        y="real_skew",
        kind="scatter",
        colormap=blues_cmap,
        colorbar=False,
        c=run_metadata["epoch"],
        alpha=0.5,
        s=scatter_size,
    )
    run_metadata.plot(
        ax=ax4,
        x="epoch",
        y="fake_skew",
        kind="scatter",
        colormap=oranges_cmap,
        colorbar=False,
        c=run_metadata["epoch"],
        alpha=0.5,
        s=scatter_size,
    )

    # plot real and fake kurtosis
    run_metadata.plot(
        ax=ax5,
        x="epoch",
        y="real_kurtosis",
        kind="scatter",
        colormap=blues_cmap,
        colorbar=False,
        c=run_metadata["epoch"],
        alpha=0.5,
        s=scatter_size,
    )
    run_metadata.plot(
        ax=ax5,
        x="epoch",
        y="fake_kurtosis",
        kind="scatter",
        colormap=oranges_cmap,
        colorbar=False,
        c=run_metadata["epoch"],
        alpha=0.5,
        s=scatter_size,
    )

    # plot learning curves
    run_metadata.plot(
        ax=ax6, x="epoch", y="d_real_error", color="black", alpha=0.5
    )
    run_metadata.plot(
        ax=ax6, x="epoch", y="d_fake_error", color="red", alpha=0.4
    )
    run_metadata.plot(ax=ax6, x="epoch", y="g_error", color="blue", alpha=0.4)

    for ax in [ax2, ax3, ax4, ax5, ax6]:
        ax.set_xlabel("epoch (1k)")


def plot_gan(output_dir):
    run_metadata = pd.read_csv(output_dir / "gan_gaussian_metadata.csv")
    run_samples = pd.read_csv(output_dir / "gan_gaussian_samples.csv")

    fig, gs, ax1, ax2, ax3, ax4, ax5, ax6 = create_grid_spec(
        figsize=(16, 10), wspace=0.25, hspace=3
    )

    # plot final distribution
    sample_columns = get_sample_columms(run_samples)
    real_samples = run_samples.query("sample_type == 'real'")[sample_columns]
    fake_samples = run_samples.query("sample_type == 'fake'")[sample_columns]
    plot_axes(
        real_samples.iloc[-1],
        fake_samples.iloc[-1],
        run_metadata,
        ax1,
        ax2,
        ax3,
        ax4,
        ax5,
        ax6,
    )

    ax1.legend(["real samples", "fake samples"])
    ax2.set_ylabel("mean")
    ax3.set_ylabel("stddev")
    ax4.set_ylabel("skew")
    ax5.set_ylabel("kurtosis")
    ax6.set_ylabel("error")
    ax6.legend(["D real error", "D fake error", "G error"])
    sns.despine()
    fig.savefig(output_dir / "gan_gaussian_plot.png")
