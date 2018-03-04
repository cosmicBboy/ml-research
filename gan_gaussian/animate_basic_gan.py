"""Script to animate GAN generative samples."""

import pandas as pd
import matplotlib.animation as animation
import seaborn as sns

from visualize_basic_gan import plot_axes, create_grid_spec, get_sample_columms

dpi = 300


def animate_gan_distribution():
    sns.set_style("white")
    run_metadata = pd.read_csv("gan_gaussian_metadata.csv")
    run_samples = pd.read_csv("gan_gaussian_samples.csv")
    sample_columns = get_sample_columms(run_samples)
    all_samples = run_samples[sample_columns]
    fake_data = run_samples.query("sample_type == 'fake'")
    fake_samples = fake_data[sample_columns]
    real_samples = run_samples.query("sample_type == 'real'")[sample_columns]
    xrange = (all_samples.unstack().min(), all_samples.unstack().max())
    yrange = (0, 4000)
    epochs = fake_data["epoch"].values
    N = fake_data.shape[0]

    fig, gs, ax1, ax2, ax3, ax4, ax5, ax6 = create_grid_spec(
        figsize=(16, 10), wspace=0.25, hspace=3)

    def update_img(n, fake_samples, real_samples, epochs):
        epoch = epochs[n]
        metadata = run_metadata[run_metadata.epoch <= epoch]
        for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
            ax.clear()
        plot_axes(
            real_samples.iloc[n], fake_samples.iloc[n], metadata,
            ax1, ax2, ax3, ax4, ax5, ax6, scatter_size=20)
        ax1.set_xlim(xrange)
        ax1.set_ylim(yrange)
        ax1.annotate(
            "epoch %s" % epoch, xy=(0.025, 0.975),
            xycoords="axes fraction", fontweight="bold")
        ax1.legend(["real samples", "fake samples"])
        ax2.set_ylabel("mean")
        ax3.set_ylabel("stddev")
        ax4.set_ylabel("skew")
        ax5.set_ylabel("kurtosis")
        ax6.set_ylabel("error")
        ax6.legend(["D real error", "D fake error", "G error"])

    sns.despine()
    ani = animation.FuncAnimation(
        fig, update_img, N, interval=400,
        fargs=(fake_samples, real_samples, epochs))
    writer = animation.writers["ffmpeg"](fps=20)

    ani.save("gan_gaussian_evolution.mp4", writer=writer, dpi=dpi)


if __name__ == "__main__":
    animate_gan_distribution()
