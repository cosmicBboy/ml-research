"""Module for creating a structured algorithm environment.

This module extends algorithm_env.AlgorithmEnv to include a specific subset
of sklearn Estimators, primarily based on the auto-sklearn paper:

papers.nips.cc/paper/5872-efficient-and-robust-automated-machine-learning.pdf
"""

from . import components

lr = components.classifiers.logistic_regression()
import ipdb; ipdb.set_trace()
