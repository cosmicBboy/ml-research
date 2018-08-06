# DeepCASH

**Using a Sequence Models to Propose Machine Learning Frameworks for AutoML**

DeepCASH is an artificial neural net architecture for parameterizing the API
of machine learning software, in this case the [sklearn API][sklearn], in
order to select a complete machine learning pipeline in an end-to-end fashion,
from raw data representation, imputation, normalizing, feature representation,
and classification/regression.


# Why?

As the diversity of data and machine learning use cases increases, we also need
to accelerate and scale the process of training performant machine learning
systems. We'll need tools that are adaptable to the problem domain, and the
nature of the dataset, such as structured, unstructured, or semi-structured
data.


# Relevant Work

The Combined Algorithm Selection and Hyperparameter optimization
([CASH][autosklearn]) problem is an important one to solve if we want to
effectively scale and deploy machine learning systems in real-world use cases,
which often deals with small (< 10 gb) to medium size (10 - 100 gb) data.

CASH is the problem of searching through the space of all ML frameworks,
defined as an Algorithm `A` and a set of relevant hyperparameters `lambda`
and proposing a set of models that will perform well given a dataset and
a task e.g.

```
.----------.    .--------------.    .-----.    .---------------------.
| Raw Data | -> | Handle Nulls | -> | PCA | -> | Logistic Regression |
.----------.    .--------------.    .-----.    .---------------------.
```

In order to solve this problem, previous work like [autosklearn][autosklearn]
uses a Bayesian Optimization techniques [SMAC][smac] with an offline meta-
learning "warm-start" step using euclidean distance to reduce the search space
of ML frameworks. This meta-learning step was done by representing the datasets
with metadata features (e.g. number of features, skew, mean, variance, etc.) to
learn representations of the data space that perform well with respect to the ML
framework selection task.

[Neural Architecture Search][neuralarchsearch] is another approach to the CASH
problem, where a Controller network proposes "child" neural net architectures
that are trained on a training set and evaluated on a validation set, using the
validation performance `R` as a reinforcement learning reward signal to learn
the best architecture proposal policy.


# Contributions

The contributions of the DeepCASH project are two-fold: it builds on the neural
architecture search paradigm by formalating the output space of the Controller
as a sequence of tokens conditioned on the space of possible executable
`frameworks`. The scope of this project is to define a `framework`, expressed
as a piece of Python code, which evaluates to an instantiated sklearn
[`Pipeline`][sklearn-pipeline] and can be fitted on a training set and
evaluated on a validation set of a particular dataset `D`.

Following the Neural Architecture scheme, DeepCASH uses the REINFORCE algorithm
to compute the policy gradient used to update the Controller in order to learn a
policy for proposing good `frameworks` that are able to achieve high validation
set performance.

The second contribution of this project is that it proposes a conditional
ML `framework` generator by extending the Controller network to have an `encoder`
network that takes as input metadata about the dataset `D` (e.g. number of
instances, number of features). The output of the `encoder` network would be
fed into the `decoder` network, which proposes an ML `framework`. Therefore,
we can condition the output of the `decoder` network metadata on `D` to propose
customized `frameworks`.


# High-level Approach

There are two general approaches to take, with substantial tradeoffs to
consider:

## Approach 1: Character-level Controller

Generate an ML `frameworks` at the character level, such that the goal is to
output Python code using a softmax over the space of valid characters, e.g.
`A-Z`, `a-z`, `0-9`, `()[]=`, etc.

This approach requires building in fewer assumptions into the AutoML system,
however the function that the Controller needs to learn would be much more
complex: it needs to (a) generate valid sklearn code character-by-character,
and (b) generate performant algorithm and hyperparameter combinations over the
distribution of datasets and tasks.

## Approach 2: Domain-specific Controller

Generate ML `frameworks` over a state space of algorithms and hyperparameter
values, in this case, over the estimators/transformers of the `sklearn` API.

This approach requires building in more assumptions into the AutoML system,
e.g. expicitly specifying the algorithm/hyperparamater space to search over
and how to interpret the output of the Controller so as to fit a model, but
the advantage is that the Controller mainly has to learn a function that
generates performant algorithm and hyperparameter combinations.

In the DeepCash project, a `CashController` represents the policy approximator,
which selects actions based on a tree-structured set of softmax classifiers,
each one representing some part of the algorithm and hyperparameter space.
The controller selects estimators/transformers and hyperparameters in a
pre-defined manner (interpreted as embedding priors into the architecture
of the system). The ordering is the following:

- one hot encoding
- one hot encoder hyperparameters
- imputation (e.g. mean, median, mode)
- imputer hyperparameters
- rescaling (e.g. min-max, mean-variance)
- rescaler hyperparameters
- feature preprocessing (e.g. PCA)
- feature processor hyperparameters
- classification/regression
- classifier/regressor hyperparameters


# Extensions

An extension to the `encoder` would be to generalize the metadata feature
representation from hand-crafted features (e.g. mean of means of numerical
features) and instead formulate `encoder` as a sequence model, where the input
is a sequence of sequences. The first sequence contains data points or
`instances` of the dataset, the second sequence contains minimally preprocessed
`features` of that particular `instance` (note that the challenge here is how
to represent categorical features across difference datasets). The weights in
the `encoder` are trained jointly as part of the gradient computed using the
REINFORCE algorithm.


# Roadmap: Milestones

- [X] implementation of the naive (unstructured) `AlgorithmRNN`/
  `HyperparameterRNN` that seperately predict the estimators/transformers and
  hyperparameters of the ML Framework.
- [X] basic implementation of the structured `CashController` architecture
- [X] refine `CashController` with baseline function prior such that each data
  environment maps to its own value function (in this case, the exponential
  mean of rewards per episode).
- [X] implement basic meta-RL algorithm as described here in this
  [paper][meta-rl] in particular, feed `CashController` auxiliary inputs:
  - previous reward
  - previous actions
- [ ] normalize `reward - baseline` (equivalent of advantage in this system)
  by mean-centering and standard-deviation-rescaling.
- [ ] extend meta-RL algorithm by implementing memory as a lookup table that
  maps data environments to the most recent hidden state from the same data
  environment.
- [ ] extend deep cash to support regression problems.


# Analyses

The `./analysis` subfolder contains jupyter notebooks that visualize the
performance of the cash controller over time. Currently there are 5 analyses
in the project `analysis` subfolder:
- `rnn_cash_controller_experiment_analysis.ipynb`: analyzing the output of
  running `examples/example_rnn_cash_controller.py` with static plots.
- `cash_controller_analysis.ipynb`: a basic interactive analysis
  of a single job's outputs.
- `cash_controller_multi_experiment_analysis.ipynb`: analyzes multiple
  job outputs, all assumed to have one trial (training run) per job.
- `cash_controller_multi_trail_analysis.ipynb`: analyzes the
  output of one job, but that job has multiple trials.
- `cash_controller_multi_trial_experiment_analysis.ipynb`: analyzes
  the output of multiple jobs, each with multiple trials.


[neuralarchsearch]: https://arxiv.org/abs/1611.01578
[autosklearn]: papers.nips.cc/paper/5872-efficient-and-robust-automated-machine-learning.pdf
[autosklearn-package]: https://automl.github.io/auto-sklearn/stable/
[autosklearn-supp]: http://ml.informatik.uni-freiburg.de/papers/15-NIPS-auto-sklearn-supplementary.pdf
[meta-rl]: https://arxiv.org/pdf/1611.05763.pdf
[smac]: https://www.cs.ubc.ca/~hutter/papers/10-TR-SMAC.pdf
[gru]: https://arxiv.org/pdf/1406.1078.pdf
[reinforce]: https://www.quora.com/What-is-the-REINFORCE-algorithm
[tpot]: https://github.com/EpistasisLab/tpot
[h20]: http://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html
[openml]: https://www.openml.org/
[pytorch-reinforce]: https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py
[sklearn]: http://scikit-learn.org/stable/
[sklearn-pipeline]: http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
