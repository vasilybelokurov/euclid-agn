"""Numerical housekeeping shared by the fitting code.

numpy 1.26.4 built against Apple's Accelerate BLAS - the build in this
environment - raises spurious ``RuntimeWarning: divide by zero / overflow /
invalid value encountered in matmul`` on ordinary finite matrix products.
Verified: a 600x64 by 64x600 product of standard Gaussian random numbers is
entirely finite and fires all three warnings.

Suppressing them blindly would hide genuine numerical failures, so the rule in
this package is: suppress around the matmul-heavy block, then assert on the
result with :func:`check_finite`.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from functools import wraps
from typing import Any, TypeVar

import numpy as np

F = TypeVar("F", bound=Callable[..., Any])


class NumericalError(RuntimeError):
    """A numerical routine produced a non-finite result."""


@contextmanager
def suppress_blas_warnings():
    """Ignore floating-point warnings raised by the BLAS, nothing else."""
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        yield


def blas_safe(function: F) -> F:
    """Run a function inside :func:`suppress_blas_warnings`."""

    @wraps(function)
    def wrapper(*args, **kwargs):
        with suppress_blas_warnings():
            return function(*args, **kwargs)

    return wrapper  # type: ignore[return-value]


def check_finite(name: str, array: np.ndarray) -> np.ndarray:
    """Raise if ``array`` is not entirely finite."""
    array = np.asarray(array)
    if not np.all(np.isfinite(array)):
        raise NumericalError(f"{name} contains non-finite values")
    return array
