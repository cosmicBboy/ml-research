# Research Log

## 03/28/2020

### Learning Rate Tuning

- floydhub job: [376](https://www.floydhub.com/nielsbantilan/projects/deep-cash/376)
- analysis notebook: `analysis/20200218_metalearn_a2c_openml_cc18_tune_learning_rate.ipynb`

In this experiment, I tried 4 different learning rate settings over 1000
episodes of training on binary tasks only, finding that `0.005` is the optimal
learning rate, all of the other hyperparameters held fixed. I assessed
meta-training performance with cumulative regret, and meta-test performance with
cumulative reward.

For meta-test performance, the weights of the controller were frozen, and
50 ML frameworks proposals were generated for binary classification, multiclass
classification, and regression tasks.

The optimal learning rate of `0.005` performed the best for binary target types
in the meta-test datasets with respect to cumulative reward. For regression
tasks, the controller trained with learning rate `0.0005` initially performed
better from the 0th to about the 25th shot, but after that the controller
trained with learning rate `0.005` performed better.

Based on n-shot (0- to 50-shot) learning validation scores, the meta-test reward
and validation scores for binary and multiclass tasks suggests no evidence of
meta-learning, as the controller did not demonstrate the ability to improve the
validation scores of its proposed ML frameworks over time.

However, its validation performance on the regression tasks do suggest that
it was able to improve validation scores from the 0th to the 50th shot. These
are preliminary results to be explored further. One explanation for this is that
since binary and multiclass tasks use the same underlying sklearn estimators,
it could be that the system has found the optimal estimators on average and
can do no better given the hyperparameter ranges and estimators in the
controller's action space. The action space could be expanded in three ways:

- expand the available hyperparameter settings
- expand the action space to new estimator types
- expand the action space to propose multiple estimators and ensemble them.

Another concurrent explanation would be that regression tasks require the
controller to access a fundamentally different region of its action space and
it was here that it demonstrated meta-learning ability by finding better
regressors over time.

### Entropy Coefficient Tuning

- floydhub job: [371](https://www.floydhub.com/nielsbantilan/projects/deep-cash/371)

I performed the same analysis, this time holding the learning rate fixed at
`0.005` and tuning various entropy coefficient settings. The higher this
setting, the more the controller is discouraged from pre-maturely settling on
a potentially sub-optimal set of actions.

Based on the cumulative reward on the meta-test datasets, I found mixed results,
with `0.01` performing best on binary tasks, `0.03` on multiclass tasks,
and `0.03` and `0.01` performing equally well on regression tasks. The
experiment was unable to recover the meta-learning result in the regression
validation scores from 0-50 shots of meta-test learning, so that observation
from the learning rate tuning experiment would need to be tested more
systematically.

### Training Capacity Tuning

- floydhub jobs:
  - [n episodes](https://www.floydhub.com/nielsbantilan/projects/deep-cash/388)
  - [n layers](https://www.floydhub.com/nielsbantilan/projects/deep-cash/390)
  - [n units per layer](https://www.floydhub.com/nielsbantilan/projects/deep-cash/391)

Here I performed a similar analysis, tuning three types of hyperparameters
in separate experiments: meta-training time (number of episodes), number of
layers, and layer unit size. For all of these experiments I set learning rate
to `0.005` and the entropy coefficient to `0.1`.

For these experiments I did not assess the meta-test performance of the
controller on regression tasks.

#### Training Time

Interestingly, the controller trained for 2000 episodes performs the best in
the binary meta-test datasets but not on the multiclass classification
datasets.

#### Number of Layers

Controllers with layer size = 6 performed the best on the binary meta-test
datasets but performed roughly on par with controller layer size = 3 on the
multiclass dataset.

#### Hidden Layer Size

For input, hidden, and output unit sizes, 120 units performed the best for
both binary and multiclass classification tasks.


### Next Steps

The next steps of this project has several paths ahead:

1. **To improve experimentation and iteration**, it might be good to find a
   smaller subset of datasets for each task to train and test on. Research on
   meta-learning typically trains on a small number of similarly structured
   tasks, whereas so far I've been using tens of different datasets to train
   on. Before scaling up to many datasets, it might be expedient to select
   a smaller set of datasets of varying difficulty. This would speed up
   time-to-result for any given experiment.
2. (1) would enable more conclusive experiments for the meta-learning
   capabilities of the metalearn architecture. It seems like the most promising
   direction for this would be to train classification task types (binary and
   multiclass), and then to meta-test on regression tasks, since the controller
   will have to access a previously un-traversed part of the action space.
3. Expand metafeature set to include more comprehensive set of dataset
   statistics. See [this paper][supp-efficient-robust-auto-ml] for examples
   of such statistics. This would provide the controller with a richer feature
   set to learn the meta-learning algorithm in the outer loop.
3. **To further improve performance**, extend the controller to ensemble a
   set of ML frameworks proposed in parallel per iteration.
4. **Improve inner-loop interface** so that the system is easier to maintain
   and extend. This would involve some refactoring of the way that datasets
   are handled for training in the inner loop, e.g. support text data. Make it
   more flexible to use different training and validation procedure, not just
   bootstrap sampling, i.e. also support cross-validation.


## 12/04/2019

### Single RNN Does not Meta-learn

Based on the results in `analysis/metalearn_openml_cc18_benchmark.ipynb`,
it doesn't appear that the RNN loop is able to model the previous action and
reward signal to implement an RL learning policy even with the RNN's weights
frozen. Consider two hypotheses:

1. Currently gradient tracking is reset on the previous action vector as it's
   passed along to the next time step. In a sequence decoder setting,
   does this even make sense to do? The gradient should be backpropped along all
   the way back to the beginning of the episode, so I think the current
   implementation is preventing the controller from learning time scales
   beyond selecting the actions that specify an ML Framework.
2. Right now the RNN has to model two things: the sequences of actions that
   produce high rewards, and how previous action and reward `(a_prev, r_prev)`
   maps to the current action and reward `(a_curr, r_curr)`. Perhaps two
   separate RNNs would enable meta-learning, one to model the
   `(r_prev, a_prev) -> (a_curr, r_curr)` and another to learn sequences of
   actions that specify and ML Framework that maximize the validation performance
   reward signal.

`1` is easier to implement and test, so this should be the first thing to try.

### Experiment: Try Doing Inference on Other Task Types

Looking at the openml cc18 benchmark results, it actually looks like the
controller is achieving decent validation scores, i.e. it doesn't test the
controller's ability to learn different kinds of supervised tasks of similar
structure.

One way to test this would be to train the contoller on a purely binary
classification task distribution and then generate inferences for binary,
multiclass, and regression tasks. We'd expect it to already do well for binary
tasks even in the test task distribution, but it will probably not have
explored algorithm space for multiclass or regression estimators.

## 10/01/2019

### Thoroughly Test Meta-learning Capabilities

So far, experiments have demonstrated that the ML Framework controller can
learn to propose good models on the training task distribution. However, no
experiments have been done to demonstrate that the controller implements a
reinforcement learning model within the recurrent loop of the RNN.

First need to thoroughly test the current system's metalearning capabilities:

- Train task type-specific distributions: binary, multiclass, regression
- Evaluate the out-of-distribution test tasks with frozen weights on the
  controller. Null hypothesis is that the controller displays random behavior.
  Alternative hypothesis is that controller displays increasing mean reward.

It's possible that the there are a few issues in the current architecture that
are preventing the controller from learning an RL policy in the RNN loop:

- Problem: currently, models have been trained on 16 iterations per episode,
  and each gradient update occurs only at the end of the episode. This means
  that the controller only has a chance to update itself once based on the
  aggregate loss over the entire episode (this would Monte Carlo policy
  gradient learning). At the end of an episode, the task is re-sampled.
  - Solution: loosen the MetaLearnReinforce implementation to implement
    TD-lambda learning so that gradient updates can be based on as few as one
    iteration during an episode.

- Problem: in experiments so far, the controller is a function of the previous
  reward, where the reward is not normalized. This may cause issues across
  tasks with different reward distributions.
  - Solution: normalize the reward per iteration.


## 09/05/2018

### Correct Baseline Controller Implementation

Completed analysis of experiments from floydhub jobs 150-151 looking at the
training metrics of the correct implementation of the controller. There was
a nasty bug in which the controller was sampling the softmax action
distribution twice, leading to two discrepent values: the discrete action
choice did not necessarily correspond to its correct action probability.

The analysis can be found in:
`analysis/20180809_experiments_jobs_150_151.ipynb`

### Entropy Coefficient Hyperparemeter Tuning Coefficients

Also added an analysis of experiments tuning the `entropy_coef` hyperparameter,
which modulates the extent to which the loss function penalizes action
probability distributions that have lower entropy (higher confidence in some
particular set of actions), thereby encouraging exploration.

The plots/notes can be found in:
`analysis/20180902_experiments_entropy_jobs_166-175.ipynb`

## 09/03/2018

Adding some notes on the REINFORCE algorithm (actually generalizeable to other
RL algos):

Since the REINFORCE algorithm is a gradient ascent method, negate the log
probability in order to do gradient descent on the negative expected rewards.

- If reward is positive and selected action prob is close to 1
  (-log prob ~= 0), policy loss will be positive and close to 0,
  controller's weights will be adjusted by gradients such that the
  selected action is more likely (i.e. by pushing action prob closer to 1)
- If reward is positive and selection action prob is close to 0
  (-log prob > 0), policy loss will be a large positive number,
  controller's weights will be adjusted such that selected
  action is less likely (i.e. by pushing action prob closer to 1).
- If reward is negative and selected action prob is close to 1
  (-log prob ~= 0), policy loss will be negative but close to 0,
  meaning that gradients will adjust the weights to minimize policy
  loss (make it more negative) by making the selected action
  probability even more negative (i.e. by pushing the action prob closer to 0).
- If reward is negative and selected action prob is close to 0,
  (-log prob > 0), then the policy loss will be a large negative number
  and the gradients will make the selected action prob even less likely
  (i.e. by pushing the action prob closer to 0).

## 08/06/2018

Need to run the following experiments:
- Train controller with the mean-centered, std-rescaled `reward - baseline`.
- Train controller using negative rewards for ML framework proposals that fail
  to fit (try various values, `[-0.1, -0.25, -0.5, -1]`)

There are a couple of tuning experiments to try as well:
- `learning_rate`
- `number rnn layers`
- `number hidden units`


## 07/30/2018

Adding notes on research ideas:
- try modifying the baseline function to be dependent on the data env, i.e.
  maintain an exponential mean for each data environment. The problem with the
  current implementation is that the baseline function is an exponential mean
  of the previous MLF validation performances across all data environments
  sampled by the task environment. This might be distorting the reward signal
  since we compute `reward` - `baseline`, which is a simplified version of the
  advantage function in the actor-critic RL framework. We want to use a
  baseline function conditioned on the data environment.
- mean-center and normalize by standard deviation of the reward, as seen here:
  https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py#L6
- implement the meta-RL algorithm as described here in this
  [paper](https://arxiv.org/pdf/1611.05763.pdf) (NIPS 2017 video
  [here](https://www.youtube.com/watch?v=9kQ-6VcdtNQ)).
  - RNN takes in auxiliary inputs:
    - previous reward r_{t-1}
    - previous actions. NOTE: in the paper it's really the previous action
      a_{t-1}, but in the deep cash context, it could be implemented as
      a set of actions from the previously proposed MLF, or the activation
      of the last unrolled layer of the RNN from t-1)
    - implement memory of past exploration through a simple lookup table
      of contexts (data envs) and their most recent hidden state (the hidden
      state of the RNN at the last time step).
    - idea: extend the memory functionality to ensemble MLFs proposed by
      running the controller RNN using the hidden states from the
      top `n` most similar contexts (similarity measured by knn)

## 07/01/2018

There have been a few developments since the last log entry, the main one being
a complete re-implementation of the controller agent. The new architecture
is a more faithful adaptation of the
[neural architecture search paper](https://arxiv.org/pdf/1611.01578.pdf).

The `metalearn` library now implements a `MetaLearnController`, which specifies an
end-to-end architecture that proposes both the estimators/transformers and
hyperparameters that compose together to form an ML framework (MLF). This
architecture is a GRU recurrent neural network that has a special decoder
that has access to the `AlgorithmSpace` from which to choose
estimators/transformers and their corresponding hyperparameters (actions).

The `MetaLearnReinforce` module, implements the REINFORCE policy gradient algorithm,
which learns the actions that maximize the expected reward based on the
observed rewards obtained from a batch of proposed MLFs. A baseline function
is used for the purposes of regularization in order to reduce the variance of
the learned policy.

Need to add units tests to be confident about the implementation of the
architecture, since it seems like the 4 toy classification datasets that the
controller currently has access to are trivial problems to solve, where the
controller is able to propose MLFs that achieve 95 - 100%
[f1-scores](https://en.wikipedia.org/wiki/F1_score).

### Learning Rate Tuning

I ran several experiments on floydhub to get more intuition on the behavior
of the controller. Based on these
[learning curves](https://www.floydhub.com/nielsbantilan/projects/deep-cash/68),
it looks like a learning rate of `0.0025` leads to learning behavior with a
particular signature:

- the `mean_reward` over episodes steadily increases over time, accompanied
  by an increase in the `loss`, which is the negative log of expected rewards
  based on a batch of proposed MLFs action log probabilities.
- on further training, the `loss` fluctuates around `0`, with a corresponding
  fluctuation of `mean_rewards`.
- as training proceeds, `n_unique_mlfs` (the number unique MLFs proposed by
  the controller) decreases, suggesting that the controller converges on a
  few MLFs that it consistently proposes.
- by the end to training (`1000` episodes), the controller's behavior seems
  quite erratic, with certain episodes having a `mean_reward` close to `-1`
  (all proposed MLFs throw an exception).

I also tried a few other learning rates, essentially any setting lower than
`0.0025` displays a fairly flat learning curve, and settings higher than
`0.0025` displaying the erratic behavior described above.

- [learning rate=0.001](https://www.floydhub.com/nielsbantilan/projects/deep-cash/60)
- [learning rate=0.005](https://www.floydhub.com/nielsbantilan/projects/deep-cash/46)


### Next Steps

- Articulate a clear set of hypotheses around the expected behavior of the
  controller, and the ideal behavior of the controller, e.g. "`mean_reward`
  and `loss` should converge to a stable value".
- Need to come up with a way of concisely visualizing the performance of the
  controller.
- Add more datasets to the task environment. It seems that the controller is
  proposing performant MLFs from the beginning of training, it might be that
  introducing more challenging datasets will change the learning curve profile
  to have more stable convergence properties.


## 05/29/2018

Need to re-think how algorithms/hyperparameters are being chosen. Currently,
there are two separate controllers for algorithms (sklearn Transformers
or Estimators) and hyperparameters. For both of these controllers, the RNN
must choose from a softmax classifier over the entire state space. This seems
inefficient because for a particular hyperparameter setting, the number of
valid hyperparameter values is small, which makes it hard for the controller
to pick the correct one.

From https://arxiv.org/pdf/1611.01578.pdf, it seems like each hyperparameter
for each RNN time-step is selected from a different softmax classifier. The
implementation specifics are not entirely clear from the paper, but it appears
that from these [slides](http://rll.berkeley.edu/deeprlcoursesp17/docs/quoc_barret.pdf)
there is a different softmax classifier for each hyperparameter, e.g:

- filter height = [1, 3, 5, 7]
- filter width = [1, 3, 5, 7]
- stride height = [1, 2, 3]
- stride width = [1, 2, 3]
- number of filters = [24, 36, 48, 64]

It looks like these variable-length softmax outputs are then transformed via an
embedding layer (to standardize the dimensions of the hyperparameter
prediction) and be fed into the RNN as input in the next time step.

It might make sense then to create an entirely different architecture for
metalearn, such that the hyperparameter controller sequence is indexed such
that a different softmax classifier predicts the state-space for a particular
hyperparameter setting, and a corresponding embedding layer is used to
standardize the dimensionality so that the softmax output can be fed into
subsequent layers.

In [this implementation](https://github.com/titu1994/neural-architecture-search/blob/master/controller.py#L287)
of the NAS RNN controller, each timestep gets it's own classifier, based on the
state space of the hyperparameter that the controller is trying to predict at
that time step.


[supp-efficient-robust-auto-ml]: https://ml.informatik.uni-freiburg.de/papers/15-NIPS-auto-sklearn-supplementary.pdf
