# Research Log

## 07/30/2018

Adding notes on research ideas:
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

The `deep_cash` library now implements a `CASHController`, which specifies an
end-to-end architecture that proposes both the estimators/transformers and
hyperparameters that compose together to form an ML framework (MLF). This
architecture is a GRU recurrent neural network that has a special decoder
that has access to the `AlgorithmSpace` from which to choose
estimators/transformers and their corresponding hyperparameters (actions).

The `CASHReinforce` module, implements the REINFORCE policy gradient algorithm,
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
deep_cash, such that the hyperparameter controller sequence is indexed such
that a different softmax classifier predicts the state-space for a particular
hyperparameter setting, and a corresponding embedding layer is used to
standardize the dimensionality so that the softmax output can be fed into
subsequent layers.

In [this implementation](https://github.com/titu1994/neural-architecture-search/blob/master/controller.py#L287)
of the NAS RNN controller, each timestep gets it's own classifier, based on the
state space of the hyperparameter that the controller is trying to predict at
that time step.
