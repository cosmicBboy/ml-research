"""Custom errors."""

import re

from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning


# TODO: generalize this to be regex expressions.
# TODO: characterize the root of each of these error messages.
# sklearn error messages that yield negative reward
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
    (ValueError, "Solver lbfgs supports only dual=False, got dual=True"),
    (ValueError, "Solver lbfgs supports only l2 penalties, got l1 penalty"),
    (ValueError, "Solver newton-cg supports only l2 penalties"),
    (ValueError, "Solver newton-cg supports only dual=False, got dual=True"),
    (ValueError, "Solver sag supports only dual=False, got dual=True"),
    (ValueError, "Solver sag supports only l2 penalties, got l1 penalty"),
    (ValueError, "Solver saga supports only dual=False, got dual=True"),
    (ValueError, "Solver liblinear does not support a multinomial backend"),
    # FeatureAgglomeration
    (ValueError,
        "The condensed distance matrix must contain only finite values"),
    (ValueError, "n_components must be < n_features"),
    (ValueError, "max_features must be in \(0, n_features\]"),
    (ValueError,
        "Unsupported set of arguments: The combination of penalty='l1' and "
        "loss='logistic_regression' are not supported when dual=True"),
    (FloatingPointError, "overflow encountered in exp"),
    (FloatingPointError, "underflow encountered in square"),
    (FloatingPointError, "divide by zero encountered in true_divide"),
    (FloatingPointError, "invalid value encountered in true_divide"),
    (FloatingPointError, "invalid value encountered in sqrt"),
    (FloatingPointError, "invalid value encountered in reduce"),
    (FloatingPointError, "underflow encountered in exp"),
    (TypeError, "'str' object is not callable"),
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
    (ImportWarning,
        "can't resolve package from __spec__ or __package__, falling back on "
        "__name__ and __path__"),
    (UserWarning, "n_components > n_samples. This is not possible."),
    (UserWarning, "n_components is too large: it will be set to"),
    (UserWarning, "Ignoring n_components with whiten=False."),
    (UserWarning,
        "FastICA did not converge. Consider increasing tolerance or "
        "the maximum number of iterations."),
    (RuntimeWarning,
        "numpy.dtype size changed, may indicate binary incompatibility"),
]


PREDICT_ERROR_MESSAGES = [
    (FloatingPointError, "overflow encountered in exp"),
    (FloatingPointError, "underflow encountered in exp"),
    (FloatingPointError, "divide by zero encountered in log"),
    (FloatingPointError, "divide by zero encountered in true_divide"),
]

PREDICT_ERROR_TYPES = tuple(set([i for i, _ in PREDICT_ERROR_MESSAGES]))

SCORE_WARNINGS = [
    (UndefinedMetricWarning,
        "F-score is ill-defined and being set to 0.0 in labels with no "
        "predicted samples")
]


class NoPredictMethodError(Exception):
    pass


class ExceededResourceLimitError(Exception):
    pass


def _is_valid_error(error, error_message_tuples):
    error_str = str(error)
    for etype, msg in error_message_tuples:
        if isinstance(error, etype) and re.match(msg, error_str):
            return True
    return False


def is_valid_fit_error(error):
    return _is_valid_error(error, FIT_ERROR_MESSAGES)


def is_valid_predict_error(error):
    return _is_valid_error(error, PREDICT_ERROR_MESSAGES)