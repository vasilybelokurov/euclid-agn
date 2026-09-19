"""Local continuum basis.

Version 1 is a regularised cubic B-spline in wavelength, not a stellar
population decomposition.  Chilingarian et al. needed SSP fitting because SDSS
optical spectra are full of stellar absorption over a wide baseline; the Euclid
red grism covers 1.19-1.90 micron at R ~ 450, where a flexible smooth function
is both sufficient and far cheaper.  A population model is a later refinement to
be justified by residuals, not assumed up front.

The basis is returned as a design-matrix block so the continuum solves jointly
with the line amplitudes in one constrained linear system.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import BSpline


def _knot_vector(wavelength: np.ndarray, n_knots: int, degree: int) -> np.ndarray:
    """Clamped knot vector with ``n_knots`` interior breakpoints.

    Breakpoints are placed at quantiles of the *fitted* wavelengths, so masked
    gaps do not leave unconstrained spline segments.
    """
    lo, hi = float(wavelength[0]), float(wavelength[-1])
    if n_knots > 0:
        quantiles = np.linspace(0.0, 1.0, n_knots + 2)[1:-1]
        interior = np.quantile(wavelength, quantiles)
        interior = np.unique(interior)
        interior = interior[(interior > lo) & (interior < hi)]
    else:
        interior = np.empty(0)
    return np.concatenate([np.full(degree + 1, lo), interior, np.full(degree + 1, hi)])


def bspline_basis(
    wavelength: np.ndarray, n_knots: int = 8, degree: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """B-spline design matrix and its knot vector.

    Parameters
    ----------
    wavelength : ndarray
        Fitted wavelengths, Angstrom, increasing.
    n_knots : int
        Number of interior knots.  ``0`` gives a single polynomial piece of the
        requested degree.
    degree : int
        Spline degree; 3 (cubic) by default.

    Returns
    -------
    design : ndarray, shape (n_pixels, n_basis)
    knots : ndarray
    """
    wavelength = np.asarray(wavelength, dtype=np.float64)
    if wavelength.ndim != 1 or wavelength.size < degree + 1:
        raise ValueError("need at least degree+1 wavelength points")
    if not np.all(np.diff(wavelength) > 0):
        raise ValueError("wavelength must be strictly increasing")
    knots = _knot_vector(wavelength, n_knots, degree)
    design = BSpline.design_matrix(wavelength, knots, degree, extrapolate=True).toarray()
    return np.asarray(design, dtype=np.float64), knots


def difference_penalty(n_basis: int, order: int = 2) -> np.ndarray:
    """Discrete ``order``-th difference operator on spline coefficients.

    This is the penalty *structure*; its strength is set by
    ``LinearProblem.regularisation_weight`` so that one dimensionless number
    means the same thing for every object.
    """
    if n_basis <= order:
        return np.zeros((0, n_basis))
    d = np.eye(n_basis)
    for _ in range(order):
        d = np.diff(d, axis=0)
    return d


def continuum_block(
    wavelength: np.ndarray,
    n_knots: int = 8,
    degree: int = 3,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Design block, penalty structure and parameter names for the continuum.

    The penalty block has one column per continuum coefficient; callers pad it
    with zero columns for the line parameters before stacking, and set its
    strength through ``LinearProblem.regularisation_weight``.
    """
    design, _ = bspline_basis(wavelength, n_knots=n_knots, degree=degree)
    penalty = difference_penalty(design.shape[1], order=2)
    names = tuple(f"continuum_{i}" for i in range(design.shape[1]))
    return design, penalty, names


def power_law_block(
    wavelength: np.ndarray,
    indices: tuple[float, ...] = (-2.0, -1.0, 0.0, 1.0),
    pivot: float | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Design block spanning power-law continua, for quasars.

    A B-spline is the wrong nuisance continuum under a broad emission line: with
    knots every few hundred Angstrom it can follow a 5,000 km/s line and absorb
    a large part of its flux.  A quasar continuum is a power law, so a handful
    of fixed power-law columns spans what is needed with four parameters that
    *cannot* bend on the scale of a line.

    Columns are ``(lambda / pivot) ** index``; the coefficients are free in
    sign, so their combination covers a continuous range of effective slopes
    and mild curvature.  There is no smoothness penalty: the basis is already
    stiff.
    """
    wavelength = np.asarray(wavelength, dtype=np.float64)
    pivot = float(pivot if pivot is not None else np.median(wavelength))
    x = wavelength / pivot
    design = np.column_stack([x**index for index in indices])
    penalty = np.zeros((0, design.shape[1]))
    names = tuple(f"powerlaw_{index:+.1f}" for index in indices)
    return design, penalty, names


#: Continuum blocks that :func:`euclid_agn.models.forward.fit_hypothesis` can use by name.
CONTINUUM_BLOCKS = {"spline": continuum_block, "power_law": power_law_block}
