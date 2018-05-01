"""A handler for generating tasks and datasets and evaluating ml frameworks."""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def _metafeatures(X_train, y_train):
    # For now this just creates a single metafeatures:
    # number training examples
    return np.array([len(X_train)])



class TaskEnvironment(object):
    """Generates datasets associated with supervised learning tasks."""

    def __init__(self, scorer, n_samples=1000, random_state=None):
        # TODO: soon this should be a set of datasets from which to sample.
        self.n_samples = n_samples
        self.data_env = make_classification(
            n_samples=self.n_samples, n_features=50, n_informative=3,
            n_classes=2, shuffle=True, random_state=random_state)
        self.X = self.data_env[0]
        self.y = self.data_env[1]
        self.data_env_index = np.array(range(n_samples))
        self.n = len(self.data_env_index)
        self.random_state = random_state
        self.scorer = scorer
        np.random.seed(self.random_state)

    def sample(self):
        # number of training samples to bootstrap
        train_index = np.random.choice(
            self.data_env_index, self.n_samples, replace=True)
        test_index = np.setdiff1d(self.data_env_index, train_index)
        # save the test partitions for evaluation
        self.X_train = self.X[train_index]
        self.y_train = self.y[train_index]
        self.X_test = self.X[test_index]
        self.y_test = self.y[test_index]
        return _metafeatures(self.X_train, self.y_train)

    def evaluate(self, ml_framework):
        try:
            ml_framework.fit(self.X_train, self.y_train)
            try:
                pred = ml_framework.predict_proba(self.X_test)[:, 1]
            except:
                pred = ml_framework.predict(self.X_test)
            return self.scorer(self.y_test, pred) * 100  # scale to [0, 100]
        except:
            return None
