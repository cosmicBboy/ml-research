# Research Log

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
