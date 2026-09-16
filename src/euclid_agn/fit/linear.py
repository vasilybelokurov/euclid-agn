"""Constrained, regularised weighted linear least squares.

This is the inner solve of the whole pipeline.  At fixed non-linear parameters
(redshift, broad-line width and offset, narrow profile shape) the model is
linear in its amplitudes, so a Stage-1 screen costs one bounded linear solve per
hypothesis rather than a general optimisation.

The problem is

.. math::

    \\min_x \\, \\lVert W^{1/2}(Ax - y) \\rVert^2 + \\lVert \\Gamma x \\rVert^2
    \\quad \\text{subject to} \\quad l \\le x \\le u,

with :math:`W = \\mathrm{diag}(1/\\sigma^2)`.  Emission-line amplitudes carry
``l = 0``; continuum coefficients are unbounded.  The regularisation matrix
:math:`\\Gamma` carries the smoothness penalty on the non-parametric narrow-line
profile.

Both terms are folded into one augmented design matrix, so a single call to
:func:`scipy.optimize.lsq_linear` handles bounds and regularisation together.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import lsq_linear


class LinearSolveError(RuntimeError):
    """The linear solve produced a non-finite result."""


#: Parameters within this fraction of a bound are reported as pinned.
BOUND_TOLERANCE = 1e-8

#: numpy built against Apple's Accelerate BLAS (the wheel in use here, numpy
#: 1.26.4) raises spurious "divide by zero / overflow / invalid encountered in
#: matmul" RuntimeWarnings on ordinary finite matrix products.  Verified with a
#: 600x64 by 64x600 product of Gaussian random numbers: the result is entirely
#: finite and all three warnings fire.  Suppressing them silently would be
#: unsafe, so the solver suppresses them *and* asserts that its outputs are
#: finite; see :func:`_check_finite`.
_FP_SUPPRESS = {"over": "ignore", "divide": "ignore", "invalid": "ignore"}


def _check_finite(name: str, array: np.ndarray) -> np.ndarray:
    if not np.all(np.isfinite(array)):
        raise LinearSolveError(
            f"{name} contains non-finite values; the design matrix is probably degenerate"
        )
    return array


@dataclass(frozen=True)
class LinearProblem:
    """A weighted, bounded, regularised linear least-squares problem.

    Parameters
    ----------
    design : ndarray, shape (n_pixels, n_parameters)
        Model basis evaluated on the usable pixels.
    data : ndarray, shape (n_pixels,)
        Observed flux density on those pixels.
    variance : ndarray, shape (n_pixels,)
        Per-pixel variance; must be positive and finite.
    lower, upper : ndarray, shape (n_parameters,)
        Box constraints.  Defaults are unbounded.
    regularisation : ndarray, shape (n_penalty, n_parameters), optional
        Penalty *structure* ``Gamma`` - typically a difference operator.  Its
        overall scale is set by ``regularisation_weight``, not by the entries
        of this matrix, so that the same matrix means the same thing for a
        bright object and a faint one.
    regularisation_weight : float
        Dimensionless strength of the penalty.  ``0`` disables it; ``1`` makes
        the penalty term comparable in Frobenius norm to the data term.
    names : tuple of str
        Parameter names, carried through to the solution for bookkeeping.
    """

    design: np.ndarray
    data: np.ndarray
    variance: np.ndarray
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None
    regularisation: np.ndarray | None = None
    regularisation_weight: float = 1.0
    names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        design = np.atleast_2d(np.asarray(self.design, dtype=np.float64))
        data = np.asarray(self.data, dtype=np.float64)
        variance = np.asarray(self.variance, dtype=np.float64)
        if design.shape[0] != data.size:
            raise ValueError(
                f"design has {design.shape[0]} rows but data has {data.size} points"
            )
        if variance.shape != data.shape:
            raise ValueError("variance must have the same shape as data")
        if not np.all(np.isfinite(variance)) or np.any(variance <= 0):
            raise ValueError("variance must be positive and finite on every fitted pixel")
        if not np.all(np.isfinite(design)):
            raise ValueError("design matrix contains non-finite entries")
        if not np.all(np.isfinite(data)):
            raise ValueError("data contains non-finite entries")
        n_par = design.shape[1]
        lower = (
            np.full(n_par, -np.inf)
            if self.lower is None
            else np.asarray(self.lower, dtype=np.float64)
        )
        upper = (
            np.full(n_par, np.inf)
            if self.upper is None
            else np.asarray(self.upper, dtype=np.float64)
        )
        if lower.size != n_par or upper.size != n_par:
            raise ValueError("bounds must have one entry per parameter")
        if np.any(lower > upper):
            raise ValueError("lower bound exceeds upper bound")
        if self.regularisation is not None:
            gamma = np.atleast_2d(np.asarray(self.regularisation, dtype=np.float64))
            if gamma.shape[1] != n_par:
                raise ValueError("regularisation must have one column per parameter")
            object.__setattr__(self, "regularisation", gamma)
        names = self.names or tuple(f"p{i}" for i in range(n_par))
        if len(names) != n_par:
            raise ValueError("names must have one entry per parameter")
        object.__setattr__(self, "design", design)
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "variance", variance)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "names", tuple(names))

    @property
    def n_pixels(self) -> int:
        return self.design.shape[0]

    @property
    def n_parameters(self) -> int:
        return self.design.shape[1]

    def weighted(self) -> tuple[np.ndarray, np.ndarray]:
        """Whitened design matrix and data (divide by sigma)."""
        weight = 1.0 / np.sqrt(self.variance)
        return self.design * weight[:, None], self.data * weight


@dataclass(frozen=True)
class LinearSolution:
    """Result of one constrained linear solve."""

    coefficients: np.ndarray
    model: np.ndarray
    chi2: float
    penalty: float
    n_pixels: int
    n_parameters: int
    effective_dof: float
    at_bound: np.ndarray
    covariance: np.ndarray | None
    names: tuple[str, ...] = ()
    success: bool = True
    message: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def chi2_reduced(self) -> float:
        denominator = self.n_pixels - self.effective_dof
        return float(self.chi2 / denominator) if denominator > 0 else float("nan")

    @property
    def log_likelihood(self) -> float:
        """Gaussian log-likelihood up to a constant that cancels in comparisons.

        The constant depends only on the variances, which are identical between
        the models being compared, so differences of this quantity are the
        quantity of interest.
        """
        return -0.5 * self.chi2

    def errors(self) -> np.ndarray:
        """1-sigma parameter errors, NaN where the covariance is unavailable.

        Errors are the linear-theory ones and are **not** valid for a parameter
        pinned at a bound; those entries are returned as NaN.  A broad-line flux
        pinned at zero is the common case, and the point of the pipeline is to
        treat it as a bounded quantity rather than pretend it is Gaussian.
        """
        if self.covariance is None:
            return np.full(self.coefficients.size, np.nan)
        variance = np.diag(self.covariance).astype(float).copy()
        variance[variance < 0] = np.nan
        errors = np.sqrt(variance)
        errors[self.at_bound] = np.nan
        return errors

    def value(self, name: str) -> float:
        return float(self.coefficients[self.names.index(name)])

    def error(self, name: str) -> float:
        return float(self.errors()[self.names.index(name)])


def solve(
    problem: LinearProblem,
    tol: float = 1e-12,
    max_iter: int | None = None,
    scale_columns: bool = True,
    scale_regularisation: bool = True,
) -> LinearSolution:
    """Solve a :class:`LinearProblem`.

    Parameters
    ----------
    scale_columns : bool
        Normalise each design column to unit norm before solving and undo the
        scaling afterwards.  This matters in practice: a continuum coefficient
        is O(1e-17) in flux-density units while a line flux is O(1e-16) in
        integrated units, and the raw normal matrix is badly conditioned.
        Scaling is exact for a box constraint whose bounds are 0 or infinite,
        which is the only kind this pipeline uses.
    scale_regularisation : bool
        Interpret the regularisation matrix *relative to the data term*: the
        penalty structure is normalised so that ``||Gamma||_F = ||A||_F`` and
        then multiplied by ``problem.regularisation_weight``.  Without this, a
        penalty expressed in physical units is either ignored or overwhelming
        depending only on the flux scale of the object, which is not a
        scientific choice.

    Notes
    -----
    Unbounded problems are solved by :func:`numpy.linalg.lstsq`, which is both
    faster and exact; bounded problems go to
    :func:`scipy.optimize.lsq_linear`.
    """
    with np.errstate(**_FP_SUPPRESS):
        return _solve(problem, tol, max_iter, scale_columns, scale_regularisation)


def _solve(
    problem: LinearProblem,
    tol: float,
    max_iter: int | None,
    scale_columns: bool,
    scale_regularisation: bool,
) -> LinearSolution:
    a_w, y_w = problem.weighted()
    gamma = problem.regularisation
    lower = problem.lower
    upper = problem.upper

    if scale_columns:
        scale = np.linalg.norm(a_w, axis=0)
        scale[scale <= 0] = 1.0
    else:
        scale = np.ones(problem.n_parameters)
    a_s = a_w / scale
    lower_s = np.where(np.isfinite(lower), lower * scale, lower)
    upper_s = np.where(np.isfinite(upper), upper * scale, upper)

    weight = float(problem.regularisation_weight)
    if gamma is not None and gamma.size and weight > 0:
        gamma_s = gamma / scale
        if scale_regularisation:
            norm = np.linalg.norm(gamma_s)
            if norm > 0:
                gamma_s = gamma_s * (np.linalg.norm(a_s) / norm)
        gamma_s = gamma_s * weight
        a_aug = np.vstack([a_s, gamma_s])
        y_aug = np.concatenate([y_w, np.zeros(gamma_s.shape[0])])
    else:
        gamma_s = None
        a_aug, y_aug = a_s, y_w

    bounded = np.any(np.isfinite(lower)) or np.any(np.isfinite(upper))
    if bounded:
        result = lsq_linear(
            a_aug, y_aug, bounds=(lower_s, upper_s), tol=tol, max_iter=max_iter
        )
        x_s = result.x
        success = bool(result.status > 0)
        message = str(result.message)
    else:
        x_s, *_ = np.linalg.lstsq(a_aug, y_aug, rcond=None)
        success = True
        message = "lstsq"

    _check_finite("solution", x_s)
    x = x_s / scale
    model = problem.design @ x
    residual = (problem.data - model) / np.sqrt(problem.variance)
    chi2 = float(residual @ residual)
    penalty = float(np.sum((gamma_s @ x_s) ** 2)) if gamma_s is not None else 0.0

    at_bound = _pinned(x_s, lower_s, upper_s)
    covariance = _covariance(a_aug, at_bound, scale)
    dof = _effective_dof(a_s, gamma_s, at_bound)

    return LinearSolution(
        coefficients=x,
        model=model,
        chi2=chi2,
        penalty=penalty,
        n_pixels=problem.n_pixels,
        n_parameters=problem.n_parameters,
        effective_dof=dof,
        at_bound=at_bound,
        covariance=covariance,
        names=problem.names,
        success=success,
        message=message,
        extra={"column_scale": scale},
    )


def _pinned(x: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    scale = np.maximum(np.abs(x), 1.0)
    at_lower = np.isfinite(lower) & (x - lower <= BOUND_TOLERANCE * scale)
    at_upper = np.isfinite(upper) & (upper - x <= BOUND_TOLERANCE * scale)
    return at_lower | at_upper


def _covariance(
    a_aug: np.ndarray, at_bound: np.ndarray, scale: np.ndarray
) -> np.ndarray | None:
    """Covariance of the free parameters, embedded in a full-size matrix.

    ``a_aug`` is the column-scaled augmented design matrix, so the inverse
    normal matrix is un-scaled here before it is returned.  Rows and columns of
    parameters pinned at a bound are set to NaN: their linear-theory error is
    meaningless.
    """
    free = ~at_bound
    if not np.any(free):
        return None
    a_free = a_aug[:, free]
    normal = a_free.T @ a_free
    try:
        inverse = np.linalg.inv(normal)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(inverse)):
        return None
    n = at_bound.size
    covariance = np.full((n, n), np.nan)
    index = np.flatnonzero(free)
    s = scale[free]
    covariance[np.ix_(index, index)] = inverse / np.outer(s, s)
    return covariance


def _effective_dof(
    a_w: np.ndarray, gamma: np.ndarray | None, at_bound: np.ndarray
) -> float:
    """Effective number of free parameters, ``tr[A (A^T A + G^T G)^-1 A^T]``.

    Without regularisation and without active bounds this is exactly the number
    of parameters.  Smoothing reduces it, which is why a plain parameter count
    would overstate the flexibility of the non-parametric narrow-line profile.
    """
    free = ~at_bound
    if not np.any(free):
        return 0.0
    a = a_w[:, free]
    normal = a.T @ a
    if gamma is not None and gamma.size:
        g = gamma[:, free]
        normal = normal + g.T @ g
    try:
        return float(np.trace(a @ np.linalg.solve(normal, a.T)))
    except np.linalg.LinAlgError:  # pragma: no cover - degenerate basis
        return float(np.count_nonzero(free))


def delta_chi2(null: LinearSolution, alternative: LinearSolution) -> float:
    """``chi2(null) - chi2(alternative)``: positive when the extra component helps.

    This is a *statistic*, not a significance.  The broad-line flux is bounded
    at zero and many hypotheses are scanned per object, so the asymptotic
    chi-squared distribution does not apply.  Calibrate empirically.
    """
    return float(null.chi2 - alternative.chi2)
