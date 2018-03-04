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

import pandas as pd
import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import visualize_basic_gan
import animate_basic_gan

# real data distribution params
r_mu = 4
r_sigma = 1.25

# model params
g_params = {
    "input_size": 1,  # random noise z as input to G
    "hidden_size": 50,  # generator complexity
    "output_size": 1,  # output size
}
d_params = {
    "input_size": 100,  # minibatch size, number of samples from real data
    "hidden_size": 100,  # discriminator complexity
    "output_size": 1,   # probability if real vs fake
}
num_epochs = 1000000
minibatch_size = d_params["input_size"]
optim_betas = (0.9, 0.999)  # betas for Adam
sample_interval = 1000  # get metadata and sample generator every t epochs
sample_size = 10000  # size of sample from generator during training
d_steps = 5  # number of steps to train discriminator per epoch
g_steps = 1  # number of steps to train generator per epoch


class Generator(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        """Initialize generator.

        This function takes as input some noise from z_sampler of size
        `input_size` and a scalar that is supposed to mimic a sample from
        r_sampler.

        :param int input_size: shape of z-noise sampler.
        :param int hidden_size: shape of hidden layer, corresponding to
            complexity of generator.
        :param int output_size: shape of data that generator is trying to learn
            how to fake.
        """
        super(Generator, self).__init__()
        self.input = nn.Linear(input_size, hidden_size)
        self.hidden = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.elu(self.input(x))  # exponential linear unit in input layer
        x = F.sigmoid(self.hidden(x))  # sigmoid non-linearity, range from 0-1
        return F.elu(self.output(x))  # output sample, same dimensions R


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
        super(Discriminator, self).__init__()
        self.input = nn.Linear(input_size, hidden_size)
        self.hidden = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.elu(self.input(x))  # exponential linear unit in input layer
        x = F.elu(self.hidden(x))  # exponential linear unit in input layer
        return F.sigmoid(self.output(x))  # output probability real/fake


def get_r_sampler(mu, sigma):
    """Sample real data from a normal distribution."""
    return lambda n: torch.Tensor(np.random.normal(mu, sigma, (1, n)))


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


def add_variance(data):
    """Add variance as feature."""
    mean = torch.mean(data.data, 1, keepdim=True)
    mean_broadcast = torch.mul(torch.ones(data.size()), mean.tolist()[0][0])
    diffs = torch.pow(data - Variable(mean_broadcast), 2.0)
    return torch.cat([data, diffs], 1)


def add_variance_input_dim_modifier(x):
    """Modify discriminator input dim if using `add_variance` preprocessor."""
    return x * 2


def get_metadata(
        epoch, d_real_error, d_fake_error, g_error, d_real_data, d_fake_data):
    d_real_data_stats = compute_stats(d_real_data)
    d_fake_data_stats = compute_stats(d_fake_data)
    return [
        epoch,
        extract(d_real_error)[0],   # loss of D on real data
        extract(d_fake_error)[0],   # loss of D on fake data
        extract(g_error)[0],        # loss of G on fooling D
        d_real_data_stats["mean"],  # mu of real data
        d_real_data_stats["std"],   # sigma of real data
        d_real_data_stats["skew"],  # skew of real data
        d_real_data_stats["kurtosis"],  # kurtosis of real data
        d_fake_data_stats["mean"],  # mu of fake data
        d_fake_data_stats["std"],   # sigma of fake data
        d_fake_data_stats["skew"],  # skew of real data
        d_fake_data_stats["kurtosis"],  # kurtosis of fake data
    ]


def print_metadata(metadata):
    msg = "Epoch {}:\nD: {:.5f}/{:.5f}" \
          "\nG: {:.5f}" \
          "\nreal stats: mu = {:.5f} sigma = {:.5f} " \
          "skew = {:.5} kurt = {:.5}" \
          "\nfake stats: mu = {:.5f} sigma = {:.5f} " \
          "skew = {:.5} kurt = {:.5}\n"
    print(msg.format(*metadata))


def train_gan(
        r_sampler, z_sampler, G, D, loss, d_opt, g_opt, num_epochs=30000,
        preprocess=add_variance):
    """Train the gan over some number of epochs."""
    run_metadata = []
    run_samples = []
    for epoch in range(num_epochs + 1):
        # train discriminator on real/fake classification task
        for d_index in range(d_steps):
            # training on real data
            D.zero_grad()  # set gradients to zero
            # sample from real data sampler, number of samples are the input
            # size of the minibatch size of the discriminator.
            d_real_data = Variable(r_sampler(d_params["input_size"]))
            d_real_prediction = D(preprocess(d_real_data))
            d_real_error = loss(d_real_prediction, Variable(torch.ones(1)))
            # compute and store gradients, but don't mutate parameters
            d_real_error.backward()

            # training on fake data
            # sample from z noise sampler, number of samples are the input
            # size of the discriminator.
            d_z_input = Variable(
                z_sampler(minibatch_size, g_params["input_size"]))
            # call .detach() to make sure we don't train G on these labels.
            d_fake_data = G(d_z_input).detach()
            d_fake_prediction = D(preprocess(d_fake_data.t()))
            d_fake_error = loss(d_fake_prediction, Variable(torch.zeros(1)))
            # run backward pass and only update parameters of D
            d_fake_error.backward()
            d_opt.step()
        # train generator to produce data that causes D to make incorrect
        # predictions (label fake data as real)
        for g_index in range(g_steps):
            G.zero_grad()
            g_z_input = Variable(
                z_sampler(minibatch_size, g_params["input_size"]))
            g_fake_data = G(g_z_input)
            # generate real/fake predictions using D
            g_fake_d_prediction = D(preprocess(g_fake_data.t()))
            # pretend that the fake samples are real, so that the loss is
            # higher when G was unable to fool D, i.e. loss is higher if
            # g_fake_d_prediction is 0 (since the second argument is a vector
            # of all 1s)
            g_error = loss(g_fake_d_prediction, Variable(torch.ones(1)))
            # run backward pass and only update parameters of G
            g_error.backward()
            g_opt.step()
        if epoch % sample_interval == 0:
            metadata = get_metadata(
                epoch, d_real_error, d_fake_error, g_error, d_real_data,
                d_fake_data)
            run_metadata.append(metadata)
            print_metadata(metadata)
            # sample from generator
            sample_fake = G(Variable(
                z_sampler(sample_size, g_params["input_size"]))).detach()
            sample_real = Variable(r_sampler(sample_size))
            run_samples.append(["fake", epoch] + extract(sample_fake))
            run_samples.append(["real", epoch] + extract(sample_real))
    run_metadata = pd.DataFrame(
        run_metadata, columns=[
            "epoch", "d_real_error", "d_fake_error", "g_error",
            "real_mu", "real_sigma", "real_skew", "real_kurtosis",
            "fake_mu", "fake_sigma", "fake_skew", "fake_kurtosis"])
    sample_columns = ["sample_%s" % i for i in range(sample_size)]
    run_samples = pd.DataFrame(
        run_samples, columns=["sample_type", "epoch"] + sample_columns)
    return run_metadata, run_samples


def main():
    """Train a generative adversarial network and collect metadata."""
    print("Using data and variances")
    r_sampler = get_r_sampler(r_mu, r_sigma)
    z_sampler = get_z_sampler()
    G = Generator(**g_params)
    D = Discriminator(
        input_size=add_variance_input_dim_modifier(d_params["input_size"]),
        hidden_size=d_params["hidden_size"],
        output_size=d_params["output_size"])
    # binary cross entropy loss: http://pytorch.org/docs/nn.html#bceloss
    loss = nn.BCELoss()
    d_opt = optim.Adam(D.parameters(), lr=2e-4, betas=optim_betas)
    g_opt = optim.Adam(G.parameters(), lr=2e-4, betas=optim_betas)
    run_metadata, run_samples = train_gan(
        r_sampler, z_sampler, G, D, loss, d_opt, g_opt, num_epochs=num_epochs)
    # visualize these data by running `python visualize_basic_gan.py`
    run_metadata.to_csv("basic_gan_metadata.csv", index=False)
    run_samples.to_csv("basic_gan_samples.csv", index=False)
    visualize_basic_gan.plot_gan()
    animate_basic_gan.animate_gan_distribution()


if __name__ == "__main__":
    main()
