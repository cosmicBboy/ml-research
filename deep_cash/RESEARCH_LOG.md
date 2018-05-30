# Research Log

05/29/2018
----------

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
