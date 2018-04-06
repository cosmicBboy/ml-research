# Sequence-model AutoML (SeqAutoML)

**Using a Sequence Models to Propose Machine Learning Frameworks for AutoML**


# Relevant Work

The Combined Algorithm Selection and Hyperparameter optimization
([CASH][autosklearn]) problem is an important one to solve if we want to
effectively scale and deploy machine learning systems in real-world use cases.
CASH is the problem of searching through the space of all ML frameworks,
defined as an Algorithm `A` and a set of relevant hyperparameters `lambda`
and proposing a set of models that will perform well given a dataset and
a task e.g.

`Raw Data -> Handle Null Values -> PCA -> Logistic Regression(hyperparemeters)`

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

The contributions of the SeqAutoML project are two-fold: it builds on the neural
architecture search paradigm by formalating the output space of the Controller
as a sequence of characters conditioned on the space of possible executable
`frameworks`. The scope of this project is to define a `framework`, a piece of
Python code, which evaluates to an instantiated sklearn `Estimator` and can
be fitted on a training set and evaluated on a validation set of a particular
dataset `D`.

Following the Neural Architecture scheme, SeqMetaML uses the REINFORCE algorithm
to compute the policy gradient used to update the Controller in order to learn a
policy for proposing good `frameworks` that are able to achieve high validation
set performance.

The second contribution of this project is that it proposes a conditional
`framework` generator by extending the Controller network to have an `encoder`
RNN that takes as input metadata about the dataset `D` (e.g. number of
instances, number of features). The vector at the end of this `encoder` network
would be used as the initial state of the `decoder` network, which generates a
string representing the proposed `framework`. Therefore, we can condition the
output of the `decoder` network metadata on `D` to propose customized
`frameworks`.

## Notes

- The `hyperparameter` tuple can be a string of characters, or a softmax of
  the hyperparameter state space encoded as a categorical variable. In the
  the softmax case, the Controller alternates between `hyperparameter_name`
  and `hyperparameter_value` softmax output layers.
- The Controller can possibly be formulated as a GAN, where the generator
  is an RNN the outputs a string of code and the discriminator tries to predict
  whether that code (i) is executable and (ii) evaluates to an sklearn
  `Estimator`


# Extensions

An extension to the `encoder` would be to generalize the metadata feature
representation from hand-crafted features (e.g. mean of means of numerical
features) and instead formulate `encoder` as a sequence model, where the input
is a sequence of sequences the first sequence contains data points or
`instances` of the dataset, the second sequence contains the `features` of
that particular `instance`. The weights in the encoder are trained jointly as
part of the gradient computed using the REINFORCE algorithm.


# Why?

As the diversity of data and machine learning use cases increases, we also need
to accelerate and scale the process of training performant machine learning
systems. We'll need tools that are adaptable to the problem domain, and the
nature of the dataset, such as structured, unstructured, or semi-structured
data.


# Roadmap

## 1. Obtain dataset using OpenML

**estimate: 2 months**

- The [OpenML][openml] platform enables users to evaluate and track their
  experiments against a particular task. This project will create a set
  of modules that will generate the auto-ml meta-dataset by programmatically
  generating ML `frameworks` and `hyperparameters` using the sklearn API.
  - each example in the auto-ml meta-dataset is a tuple of:
    `[is_executable, creates_estimator, framework_string, hyperparameter_string]`
- Run each `framework` and `hyperparameters` string against a set of datasets
  and supervised learning tasks, available on OpenML, in order to evaluate their
  performance on the validation set.


## 2. Create proof-of-concept sequence model to generate `frameworks`

**estimate: 1 month**

To verify that such a model can generate executable code, train an RNN to
generate a string that successfully executes in the Python environment and
evaluates to an `Estimator` object.

- This RNN should be conditioned on two binary variables: `is_executable`,
  and `creates_estimator`. The input should be these two binary variables plus
  a latent vector `z` of random noise that are fully connected to an
  initialization layer, which has the same dimension as the hidden activation
  of the generator RNN. The output of this model should be tuples of
  `framework` and `hyperparameters` strings.
- Evaluate the RNN by generating random samples measuring the proportion of
  samples that are executable and the proportion of those that evaluate to
  value `Estimator` or `Pipeline` objects.
- Run each `framework` and `hyperparameters` input against a set of datasets
  and supervised learning tasks, available on OpenML, in order to evaluate their
  performance on the validation set.
- **Note** an alternative to training a character-level model is to tokenize
  the code, e.g. `sklearn.linear_model -> "sklearn", ".", "linear_model"` in
  order to reduce dimensionality of the output space.


## 3. Create the Controller model by using the pre-trained sequence model from (2)

**estimate: 1 month**

- Extend the controller RNN such that it takes as input the following
  variables:
  - `is_executable`, `creates_estimator`, latent vector `z`, and metadata
    features (see [auto-sklearn supplemental material][autosklearn-supp])
- use the proposed `frameworks` to train a model on some data `D` using the
  training set and evaluate them on the validation set.
- Use the REINFORCE algorithm to learn the policy gradient to fine-tune the
  Controller.


## 4 For some set of datasets, compare `SeqMetaML` to other methods

**estimate: 1 month**

- Specify some dataset `{D_1, D_0, ..., D_n}`
- Compare against [`autosklearn`][autosklearn-package], [`tpot`][tpot], and
  [`H20.ai`][h20].


[neuralarchsearch]: https://arxiv.org/abs/1611.01578
[autosklearn]: papers.nips.cc/paper/5872-efficient-and-robust-automated-machine-learning.pdf
[autosklearn-package]: https://automl.github.io/auto-sklearn/stable/
[autosklearn-supp]: http://ml.informatik.uni-freiburg.de/papers/15-NIPS-auto-sklearn-supplementary.pdf
[smac]: https://www.cs.ubc.ca/~hutter/papers/10-TR-SMAC.pdf
[gru]: https://arxiv.org/pdf/1406.1078.pdf
[reinforce]: https://www.quora.com/What-is-the-REINFORCE-algorithm
[tpot]: https://github.com/EpistasisLab/tpot
[h20]: http://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html
[openml]: https://www.openml.org/
