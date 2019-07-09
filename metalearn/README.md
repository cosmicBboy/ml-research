# META Learn

**M**etaRL-based **E**stimator using **T**ask-encodings for
**A**utomated machine **Learn**ing

**META Learn** is a deep learning approach to automated machine learning that
parameterizes the API of machine learning software as a sequence of actions to
select the hyperparameters of a machine learning pipeline in an end-to-end
fashion, from raw data representation, imputation, normalizing, feature
representation, and classification/regression. Currently the
[sklearn API][sklearn] is the only supported ML framework.


# Why?

As the diversity of data and machine learning use cases increases, we need
to accelerate and scale the process of training performant machine learning
systems. We'll need tools that are adaptable to specific problem domains,
datasets, and the (sometimes non-differentiable) performance metrics that we're
trying to optimize. Supervised learning of classification and regression tasks
given a task distribution of small to medium datasets provides an promising
jumping off platform for programmatically generating a reinforcement learning
environment for automated machine learning (AutoML).


# Installation

install `metalearn` library:
```
pip install -e .
```

then you can run an experiment with the `metalearn` cli.

```
# run an experiment with default values
$ metalearn run experiment
```

## Running an experiment with a configuration file

Alternatively, you can create an experiment configuration file to run
your experiment.

```
# create experiment config file
$ metalearn create config my_experiment config/local --description "my experiment"

# output:
# wrote experiment config file to config/local/experiment_2018-37-25-21:37:11_my_experiment.yml
```


edit the config file `parameters` section to the set of parameters
that you want to train on, then run the experiment with

```
$ metalearn run from-config config/local/experiment_2018-37-25-21:37:11_my_experiment.yml
```


# Relevant Work

The Combined Algorithm Selection and Hyperparameter optimization
([CASH][autosklearn]) problem is an important one to solve if we want to
effectively scale and deploy machine learning systems in real-world use cases,
which often deals with small (< 10 gb) to medium size (10 - 100 gb) data.

CASH is the problem of searching through the space of all ML frameworks,
defined as an Algorithm `A` and a set of relevant hyperparameters `lambda`
and proposing a set of models that will perform well given a dataset and
a task.

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

The contributions of the META Learn project are two-fold: it builds on the neural
architecture search paradigm by formalating the output space of the Controller
as a sequence of tokens conditioned on the space of possible executable
`frameworks`. The scope of this project is to define a `framework`, expressed
as a set of hyperparameters, that can be evaluated by a machine learning
framework, like sklearn, which evaluates to an instantiated sklearn
[`Pipeline`][sklearn-pipeline]. Once defined, it can be fitted on a training
set and evaluated on a validation set of a particular dataset `D`.

Following the Neural Architecture scheme, META Learn uses the REINFORCE algorithm
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


# Training Algorithm

The environment is a distribution of `k` supervised learning tasks, consisting
of a pair `(X, y)` of features and targets, respectively. At the beginning of
an episode, the environment samples one task and for `i` iterations produces
sample splits `(X_train, y_train, X_validation, y_validation)` drawn from the
task dataset.

The `MetaLearnController` receives the current task state `s` via metafeatures
associated the task, e.g. _# of training samples_, _# of features_,
_target type_, _#of continuous features_, _#of categorical features_, etc.

Given the task state, the controller generates ML `frameworks` over a state
space of algorithms and hyperparameter values. The controller can be viewed as a
policy approximator, which selects actions based on some pre-defined
`AlgorithmSpace`, representated as a direct acyclic graph where each node
contains a set of hyperparameter values to choose from. The controller traverses
this graph via a sequential decoder by selecting hyperparameters via softmax
classifiers, where certain actions may remove certain edges from the graph.
This enforces incompatible hyperparameter configurations.

For example, the algorithm space for a possible `sklearn` pipeline would
consist of the following components:

- categorical encoder (e.g. OneHotEncoder)
- categorical encoder hyperparameters
- imputer (e.g. SimpleImputer)
- imputer hyperparameters
- rescaler (e.g. StandardScaler)
- rescaler hyperparameters
- feature processor (e.g. PCA)
- feature processor hyperparameters
- classifier/regressor (e.g. LogisticRegression, LinearRegression)
- classifier/regressor hyperparameters

When the controller reaches a terminal node in algorithm space, the environment
evalutes the selected ML `framework` and produces a validation score that the
controller uses as a reward signal. Validation performance is calibrated such
that a better score produces higher rewards. Using the REINFORCE policy
gradient method, the controller tries to find the optimal policy that
maximizes validation performance over the task distribution.


# Analyses

The `./analysis` subfolder contains jupyter notebooks that visualize the
performance of the cash controller over time. Currently there are 5 analyses
in the project `analysis` subfolder:
- `rnn_metalearn_controller_experiment_analysis.ipynb`: analyzing the output of
  running `examples/example_rnn_metalearn_controller.py` with static plots.
- `metalearn_controller_analysis.ipynb`: a basic interactive analysis
  of a single job's outputs.
- `metalearn_controller_multi_experiment_analysis.ipynb`: analyzes multiple
  job outputs, all assumed to have one trial (training run) per job.
- `metalearn_controller_multi_trail_analysis.ipynb`: analyzes the
  output of one job, but that job has multiple trials.
- `metalearn_controller_multi_trial_experiment_analysis.ipynb`: analyzes
  the output of multiple jobs, each with multiple trials.


[neuralarchsearch]: https://arxiv.org/abs/1611.01578
[apricot]: https://github.com/jmschrei/apricot
[autosklearn]: papers.nips.cc/paper/5872-efficient-and-robust-automated-machine-learning.pdf
[autosklearn-package]: https://automl.github.io/auto-sklearn/stable/
[autosklearn-supp]: http://ml.informatik.uni-freiburg.de/papers/15-NIPS-auto-sklearn-supplementary.pdf
[meta-rl]: https://arxiv.org/pdf/1611.05763.pdf
[smac]: https://www.cs.ubc.ca/~hutter/papers/10-TR-SMAC.pdf
[gan-imputation]: http://proceedings.mlr.press/v80/yoon18a.html
[gru]: https://arxiv.org/pdf/1406.1078.pdf
[reinforce]: https://www.quora.com/What-is-the-REINFORCE-algorithm
[tpot]: https://github.com/EpistasisLab/tpot
[h20]: http://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html
[openml]: https://www.openml.org/
[pytorch-reinforce]: https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py
[sklearn]: http://scikit-learn.org/stable/
[sklearn-pipeline]: http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
[xgboost]: https://xgboost.readthedocs.io/en/latest/python/python_intro.html
