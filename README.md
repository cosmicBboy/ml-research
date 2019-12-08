# ML Research
Research projects in Machine Learning


# Environment

```
conda create -n ml-research
source activate ml-research
make deps
```

Install [`OpenML`](https://github.com/openml/openml-python)

```
git clone https://github.com/openml/openml-python
cd openml-python
python setup.py install
cd ..
rm -rf openml-python
```

Install jupyter lab extensions:

```
jupyter labextension install @jupyterlab/plotly-extension
```

# Projects

## Generative Adversarial Networks for Gaussian Distributions

The `gan_gaussian` project is a basic intuition-builder for GANs. It implements
as simple GAN that tries to learn a gaussian (normal) distribution. The project
results can be reproduced by running:

```
cd gan_gaussian
python basic_gan.py
```

Running the script produces a few artifacts that are meant to help build some
intuition about how GANs work and what kinds of distributions it learns when
the generator is trying to trick the discriminator into accepting the
generator's fake samples.

- `gan_gaussian_plot.png`
- `gan_gaussian_evolution.png`


## METAlearn

**M**etaRL-based **E**stimator using **T**ask-encodings for
**A**utomated machine **Learn**ing

**META Learn** is a deep learning approach to automated machine learning that
parameterizes the API of machine learning software as a sequence of actions to
select the hyperparameters of a machine learning pipeline in an end-to-end
fashion, from raw data representation, imputation, normalizing, feature
representation, and classification/regression. Currently the
[sklearn API][sklearn] is the only supported ML framework.