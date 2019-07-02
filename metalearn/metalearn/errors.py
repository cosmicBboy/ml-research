"""Module for handling errors occurring in the TaskEnvironment runtime.

This module handles the validity of exceptions raised by the TaskEnvironment
when fitting, predicting, and scoring a propoosed ML framework.
"""

import re

from scipy.optimize.optimize import LineSearchWarning
from scipy.linalg import LinAlgWarning
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning


# sklearn error messages that yield negative reward
# TODO: need to add unit tests for this module
# TODO: characterize the root of each of these error messages.
FIT_ERROR_MESSAGES = [
    (AttributeError,
        "module 'scipy.cluster._hierarchy' has no attribute 'nn_chain'"),
    (ValueError, "Cannot extract more clusters than samples"),
    (ValueError, "l2 was provided as affinity."),
    (ValueError,
        "l1 was provided as affinity. Ward can only work with euclidean "
        "distances"),
    (ValueError,
        "manhattan was provided as affinity. Ward can only work with "
        "euclidean distances"),
    (ValueError,
        "cosine was provided as affinity. Ward can only work with "
        "euclidean distances"),
    (ValueError, "No feature in X meets the variance threshold"),
    (ValueError,
        "The condensed distance matrix must contain only finite values"),
    (ValueError, "n_components must be < n_features"),
    (ValueError, r"max_features must be in \(0, n_features\]"),
    (ValueError,
        r"Input contains NaN, infinity or a value too large for "
        r"dtype\('float64'\)"),
    (ValueError, "Input X must be non-negative"),
    (FloatingPointError, "^overflow encountered in"),
    (FloatingPointError, "^underflow encountered in"),
    (FloatingPointError, "^divide by zero encountered in"),
    (FloatingPointError, "^invalid value encountered in"),
    (FloatingPointError, "^underflow encountered in"),
    (TypeError, "'str' object is not callable"),
    (TypeError, "'numpy.float64' object cannot be interpreted as an integer"),
    (ZeroDivisionError,
        "Current sag implementation does not handle the case step_size"),
]
FIT_ERROR_TYPES = tuple(set([i for i, _ in FIT_ERROR_MESSAGES]))


# these should be ignored during fitting time so that the mlf can still
# generate validation scores for these
FIT_WARNINGS = [
    (ConvergenceWarning,
        "The max_iter was reached which means the coef_ did not converge"),
    (ConvergenceWarning,
        "newton-cg failed to converge. Increase the number of iterations"),
    (ConvergenceWarning,
        "The max_iter was reached which means the coef_ did not converge"),
    (ConvergenceWarning,
        "Objective did not converge. You might want to increase the number of "
        "iterations. Fitting data with very small alpha may cause precision "
        "problems."),
    (ImportWarning,
        "can't resolve package from __spec__ or __package__, falling back on "
        "__name__ and __path__"),
    (LineSearchWarning, "The line search algorithm did not converge"),
    (LinAlgWarning,
        "scipy.linalg.solve\nIll-conditioned matrix detected. Result "
        "is not guaranteed to be accurate.\nReciprocal condition "
        "number"),
    (UserWarning, "Line Search failed"),
    (UserWarning, "n_components > n_samples. This is not possible."),
    (UserWarning, "n_components is too large: it will be set to"),
    (UserWarning, "Ignoring n_components with whiten=False."),
    (UserWarning,
        "FastICA did not converge. Consider increasing tolerance or "
        "the maximum number of iterations."),
    (UserWarning,
        "Singular matrix in solving dual problem. Using least-squares "
        "solution instead."),
    (RuntimeWarning,
        "numpy.dtype size changed, may indicate binary incompatibility"),
]


PREDICT_ERROR_MESSAGES = [
    (FloatingPointError, "overflow encountered in exp"),
    (FloatingPointError, "overflow encountered in multiply"),
    (FloatingPointError, "overflow encountered in power"),
    (FloatingPointError, "overflow encountered in reduce"),
    (FloatingPointError, "divide by zero encountered in log"),
    (FloatingPointError, "divide by zero encountered in true_divide"),
    (FloatingPointError, "divide by zero encountered in double_scalars"),
    (FloatingPointError, "underflow encountered in true_divide"),
    (FloatingPointError, "underflow encountered in power"),
    (FloatingPointError, "underflow encountered in reciprocal"),
    (FloatingPointError, "underflow encountered in exp"),
    (FloatingPointError, "underflow encountered in multiply"),
    # SimpleImputer algorithm component
    (ValueError,
        r"The features \[.+\] have missing values in transform "
        "but have no missing values in fit"),
    # KNearestNeighbors algorithm component
    (ValueError,
        r"^Expected n_neighbors <= n_samples,")
]

PREDICT_ERROR_TYPES = tuple(set([i for i, _ in PREDICT_ERROR_MESSAGES]))

SCORE_ERRORS = (ValueError, )
SCORE_WARNINGS = [
    (UndefinedMetricWarning,
        "F-score is ill-defined and being set to 0.0 in labels with no "
        "predicted samples"),
    (UndefinedMetricWarning,
        "F-score is ill-defined and being set to 0.0 in labels with no true "
        "samples."),
    (RuntimeWarning, "invalid value encountered in log"),
]


def _is_valid_error(error, error_message_tuples):
    error_str = str(error)
    for etype, msg in error_message_tuples:
        if isinstance(error, etype) and re.match(msg, error_str):
            return True
    return False


def is_valid_fit_error(error):
    """Return True if error in MLF fit is valid for controller training.

    The criterion for acceptability is somewhat tautological at this point,
    essentially defined as errors that result from things like poorly specified
    hyperparameters and other errors raised by the sklearn API.

    Things that are not explicitly listed in the FIT_ERROR_MESSAGES global
    variable are considered invalid fit errors. This helps to minimize
    unexpected behavior when training the CASH controller.

    :params BaseException error: any subclass of BaseException.
    :returns: True if valid fit error, False otherwise.
    """
    return isinstance(error, FIT_ERROR_TYPES) and \
        _is_valid_error(error, FIT_ERROR_MESSAGES)


def is_valid_predict_error(error):
    """Return True if error in MLF predict is valid for controller training."""
    return isinstance(error, PREDICT_ERROR_TYPES) and \
        _is_valid_error(error, PREDICT_ERROR_MESSAGES)
