"""Warnings to ignore when importing package dependencies."""

import warnings

warnings.filterwarnings(
    "ignore", message="numpy.core.umath_tests is an internal NumPy module",
    category=DeprecationWarning)
