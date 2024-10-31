#!/usr/bin/env python

"""A Basic GAN Implementation.

@author Niels Bantilan <niels.bantilan@gmail.com>

adapted from: https://github.com/devnag/pytorch-generative-adversarial-networks

GANs consist of the following components:

- R: real data distribution.
- G: the generator trying to create realistic data samples.
- Z: random noise that the generator G takes as input.
- D: the discriminator trying to distinguish real and fake data.
"""

from pathlib import Path

import click
import pandas as pd
import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import visualize_basic_gan
import animate_basic_gan

# model params
g_params = {
    "input_size": 10,  # random noise z as input to G
    "hidden_size": 100,  # generator complexity
    "output_size": 1,  # output size
}
d_params = {
    "input_size": 1000,  # minibatch size, number of samples from real data
    "hidden_size": 50,  # discriminator complexity
    "output_size": 1,  # probability if real vs fake
}
minibatch_size = d_params["input_size"]
learning_rate = 2e-5
optim_betas = (0.9, 0.999)  # betas for Adam
sample_interval = 1000  # get metadata and sample generator every t epochs
sample_size = 10000  # sample size from generator for reporting
d_steps = 1  # number of steps to train discriminator per epoch
g_steps = 1  # number of steps to train generator per epoch


class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """Initialize generator.

        This function takes as input some noise from z_sampler of size
        `input_size` and a scalar that is supposed to mimic a sample from
        data_sampler.

        :param int input_size: shape of z-noise sampler.
        :param int hidden_size: shape of hidden layer, corresponding to
            complexity of generator.
        :param int output_size: shape of data that generator is trying to learn
        """
        super().__init__()
        self.input = nn.Linear(input_size, hidden_size)
        self.hidden = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, hidden_size),
        )
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.elu(self.input(x))
        x = torch.sigmoid(self.hidden(x))
        return self.output(x)


class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """Initialize discriminator.

        This function takes as input some `input_size` number of samples and
        outputs the probability that those samples are real of fake.

        Note that the discriminator is more complex (3 hidden layers) than the
        generator. A practical tip for training GANs is to give the
        discriminator an edge over the generator, either by training it more
        often, or by using a more complex model.

        :param int input_size: shape of minibatch size.
        :param int hidden_size: shape of hidden layer, corresponding to
            complexity of discriminator.
        :param int output_size: shape of output, prediction fake/real.
        """
        super().__init__()
        self.input = nn.Linear(input_size, hidden_size)
        self.hidden = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """Outputs probability that data is real/fake."""
        x = F.elu(self.input(x))
        x = F.elu(self.hidden(x))
        return torch.sigmoid(self.output(x)).view(1)


def get_data_sampler(mus, sigmas):
    """Sample real data from a mixture of normal distributions."""

    def sampler(n):
        data = []
        for mu, sigma in zip(mus, sigmas):
            data.append(np.random.normal(mu, sigma, (1, n // len(mus))))
        data = np.concatenate(data, axis=1)
        np.random.shuffle(data)
        return torch.Tensor(data)

    return sampler


def get_z_sampler():
    """Sample fake data latent parameters z from uniform distribution."""
    return lambda m, n: torch.rand(m, n)


def extract(v):
    """Get data from pytorch Variable."""
    return v.data.storage().tolist()


def compute_stats(x):
    """Compute distribution statistics for a sample x."""
    x = extract(x)
    return {
        "mean": np.mean(x),
        "std": np.std(x),
        "skew": scipy.stats.skew(x),
        "kurtosis": scipy.stats.kurtosis(x),
    }


def get_metadata(
    epoch, d_real_error, d_fake_error, g_error, d_real_data, d_fake_data
):
    d_real_data_stats = compute_stats(d_real_data)
    d_fake_data_stats = compute_stats(d_fake_data)
    return [
        epoch,
        extract(d_real_error)[0],  # loss of D on real data
        extract(d_fake_error)[0],  # loss of D on fake data
        extract(g_error)[0],  # loss of G on fooling D
        d_real_data_stats["mean"],  # mu of real data
        d_real_data_stats["std"],  # sigma of real data
        d_real_data_stats["skew"],  # skew of real data
        d_real_data_stats["kurtosis"],  # kurtosis of real data
        d_fake_data_stats["mean"],  # mu of fake data
        d_fake_data_stats["std"],  # sigma of fake data
        d_fake_data_stats["skew"],  # skew of real data
        d_fake_data_stats["kurtosis"],  # kurtosis of fake data
    ]


def print_metadata(metadata):
    msg = (
        "Epoch {}:\ndiscriminator loss: real={:.5f}/fake={:.5f}"
        "\ngenerator loss: {:.5f}"
        "\nreal stats: mu = {:.5f} sigma = {:.5f} "
        "skew = {:.5} kurtosis = {:.5}"
        "\nfake stats: mu = {:.5f} sigma = {:.5f} "
        "skew = {:.5} kurtosis = {:.5}\n"
    )
    print(msg.format(*metadata))


def train_gan(
    data_sampler,
    z_sampler,
    generator,
    discriminator,
    loss,
    d_opt,
    g_opt,
    num_epochs=30000,
):
    """Train the gan over some number of epochs."""
    run_metadata = []
    run_samples = []
    for epoch in range(num_epochs + 1):
        # train discriminator on real/fake classification task
        for _ in range(d_steps):
            # training on real data
            discriminator.zero_grad()

            # sample from real data sampler
            d_real_data = data_sampler(d_params["input_size"])
            d_real_error = loss(discriminator(d_real_data), torch.ones(1))
            # compute and store gradients, but don't mutate parameters
            d_real_error.backward()

            # training on fake data
            # .detach() to make sure we don't train generator on these labels.
            d_fake_data = generator(
                z_sampler(minibatch_size, g_params["input_size"])
            ).detach()
            d_fake_prediction = discriminator(d_fake_data.t())
            d_fake_error = loss(d_fake_prediction, torch.zeros(1))
            # run backward pass and update parameters of disriminator
            d_fake_error.backward()
            d_opt.step()

        # train generator to produce data that causes discriminator to make
        # incorrect predictions, i.e. label fake data as real
        for _ in range(g_steps):
            generator.zero_grad()
            # pretend that the fake samples are real, so that the loss is
            # higher when generator was unable to fool discriminator, i.e. loss
            # is higher if g_fake_d_prediction is 0 since the target is a
            # vector of all 1s
            g_error = loss(
                discriminator(
                    generator(
                        z_sampler(minibatch_size, g_params["input_size"])
                    ).t()
                ),
                torch.ones(1),
            )
            # run backward pass and only update parameters of G
            g_error.backward()
            g_opt.step()

        if epoch % sample_interval == 0:
            metadata = get_metadata(
                epoch,
                d_real_error,
                d_fake_error,
                g_error,
                d_real_data,
                d_fake_data,
            )
            run_metadata.append(metadata)
            print_metadata(metadata)
            # sample from generator
            sample_fake = generator(
                z_sampler(sample_size, g_params["input_size"])
            ).detach()
            sample_real = data_sampler(sample_size)
            run_samples.append(["fake", epoch] + extract(sample_fake))
            run_samples.append(["real", epoch] + extract(sample_real))

    run_metadata = pd.DataFrame(
        run_metadata,
        columns=[
            "epoch",
            "d_real_error",
            "d_fake_error",
            "g_error",
            "real_mu",
            "real_sigma",
            "real_skew",
            "real_kurtosis",
            "fake_mu",
            "fake_sigma",
            "fake_skew",
            "fake_kurtosis",
        ],
    )
    sample_columns = ["sample_%s" % i for i in range(sample_size)]
    run_samples = pd.DataFrame(
        run_samples, columns=["sample_type", "epoch"] + sample_columns
    )
    return run_metadata, run_samples


def main(output_dir, num_epochs, mus, sigmas):
    """Train a generative adversarial network and collect metadata."""
    data_sampler = get_data_sampler(mus, sigmas)
    generator = Generator(**g_params)
    discriminator = Discriminator(
        input_size=d_params["input_size"],
        hidden_size=d_params["hidden_size"],
        output_size=d_params["output_size"],
    )
    loss = nn.BCELoss()
    d_opt = optim.Adam(
        discriminator.parameters(), lr=learning_rate, betas=optim_betas
    )
    g_opt = optim.Adam(
        generator.parameters(), lr=learning_rate, betas=optim_betas
    )
    run_metadata, run_samples = train_gan(
        data_sampler,
        get_z_sampler(),
        generator,
        discriminator,
        loss,
        d_opt,
        g_opt,
        num_epochs=num_epochs,
    )
    # visualize these data by running `python visualize_basic_gan.py`
    run_metadata.to_csv(output_dir / "gan_gaussian_metadata.csv", index=False)
    run_samples.to_csv(output_dir / "gan_gaussian_samples.csv", index=False)
    visualize_basic_gan.plot_gan(output_dir)
    animate_basic_gan.animate_gan_distribution(output_dir)


@click.command("run")
@click.argument("output_dir", type=Path)
@click.option("--num-epochs", type=int, default=10_000)
@click.option("--mu", type=float, multiple=True)
@click.option("--sigma", type=float, multiple=True)
def cli(output_dir, num_epochs, mu, sigma):
    output_dir.mkdir(exist_ok=True)
    mus = list(mu)
    sigmas = list(sigma)
    main(output_dir, num_epochs, mus, sigmas)


if __name__ == "__main__":
    cli()
