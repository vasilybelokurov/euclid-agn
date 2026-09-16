import numpy as np
import pytest

from euclid_agn.fit.linear import LinearProblem, solve
from euclid_agn.spectra.continuum import (
    bspline_basis,
    continuum_block,
    difference_penalty,
)
from euclid_agn.validation.simulator import sir_wavelength_grid


def test_basis_is_a_partition_of_unity():
    w = sir_wavelength_grid()
    design, knots = bspline_basis(w, n_knots=8, degree=3)
    assert design.shape[0] == w.size
    assert np.allclose(design.sum(axis=1), 1.0, atol=1e-10)
    assert np.all(design >= -1e-12)
    assert knots.size == design.shape[1] + 4


def test_more_knots_give_more_basis_functions():
    w = sir_wavelength_grid()
    small = bspline_basis(w, n_knots=4)[0].shape[1]
    large = bspline_basis(w, n_knots=16)[0].shape[1]
    assert large > small


def test_basis_rejects_bad_grids():
    with pytest.raises(ValueError):
        bspline_basis(np.array([1.0, 3.0, 2.0, 4.0]))
    with pytest.raises(ValueError):
        bspline_basis(np.array([1.0, 2.0]), degree=3)


def test_difference_penalty_annihilates_a_linear_ramp():
    d = difference_penalty(10, order=2)
    ramp = np.arange(10, dtype=float)
    assert np.allclose(d @ ramp, 0.0)
    assert not np.allclose(d @ (ramp**2), 0.0)
    assert difference_penalty(2, order=2).shape == (0, 2)


def test_continuum_recovers_a_smooth_spectrum():
    w = sir_wavelength_grid()
    truth = 1e-17 * (1.0 + 0.3 * np.sin((w - w[0]) / 2000.0))
    rng = np.random.default_rng(0)
    noise = 2e-19
    data = truth + rng.normal(0.0, noise, w.size)
    design, _, names = continuum_block(w, n_knots=10)
    solution = solve(
        LinearProblem(
            design=design,
            data=data,
            variance=np.full(w.size, noise**2),
            names=names,
        )
    )
    residual = (solution.model - truth) / noise
    assert np.std(residual) < 0.5
    assert solution.chi2_reduced == pytest.approx(1.0, rel=0.15)


def test_smoothness_suppresses_noise_chasing():
    w = sir_wavelength_grid()
    rng = np.random.default_rng(1)
    noise = 1e-18
    data = 1e-17 + rng.normal(0.0, noise, w.size)
    variance = np.full(w.size, noise**2)
    design, penalty, names = continuum_block(w, n_knots=60)
    rough = solve(LinearProblem(design=design, data=data, variance=variance, names=names))
    smooth = solve(
        LinearProblem(
            design=design,
            data=data,
            variance=variance,
            regularisation=penalty,
            regularisation_weight=3.0,
            names=names,
        )
    )
    assert smooth.effective_dof < rough.effective_dof
    assert np.std(np.diff(smooth.model)) < np.std(np.diff(rough.model))


def test_regularisation_weight_actually_changes_the_answer():
    """A weight that is ignored is worse than no regularisation at all."""
    w = sir_wavelength_grid()
    rng = np.random.default_rng(4)
    noise = 1e-18
    data = 1e-17 + rng.normal(0.0, noise, w.size)
    variance = np.full(w.size, noise**2)
    design, penalty, names = continuum_block(w, n_knots=40)
    dofs = []
    for weight in (0.0, 0.3, 3.0, 30.0):
        solution = solve(
            LinearProblem(
                design=design,
                data=data,
                variance=variance,
                regularisation=penalty,
                regularisation_weight=weight,
                names=names,
            )
        )
        dofs.append(solution.effective_dof)
    assert dofs == sorted(dofs, reverse=True)
    assert dofs[0] > dofs[-1] + 1.0
