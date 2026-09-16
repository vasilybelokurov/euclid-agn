"""Synthetic recovery tests for the non-parametric narrow-line system."""

from __future__ import annotations

import numpy as np
import pytest

from euclid_agn.constants import SIR_BINWIDTH_ANGSTROM
from euclid_agn.models.narrow import (
    NarrowSystem,
    fit_narrow_system,
    gaussian_profile,
    normalise_profile,
    smoothness_penalty,
    suggested_velocity_step,
    velocity_grid,
)
from euclid_agn.spectra.continuum import continuum_block
from euclid_agn.validation.simulator import (
    LineTruth,
    SpectrumTruth,
    simulate_arrays,
    sir_wavelength_grid,
)

Z = 1.20
LSF = 13.7


def system(**kwargs) -> NarrowSystem:
    base = dict(
        line_names=("Halpha", "NII6584", "NII6548"),
        z=Z,
        lsf_sigma=LSF,
        velocities=velocity_grid(1000.0, 250.0),
    )
    base.update(kwargs)
    return NarrowSystem(**base)


def test_velocity_grid_is_symmetric_and_includes_zero():
    v = velocity_grid(1000.0, 250.0)
    assert v[0] == -1000.0 and v[-1] == 1000.0
    assert 0.0 in v
    assert v.size % 2 == 1
    with pytest.raises(ValueError):
        velocity_grid(1000.0, 0.0)


def test_suggested_step_matches_the_instrumental_resolution():
    # LSF sigma 13.7 A at Halpha, z=1.2 -> lambda = 14444 A -> ~284 km/s.
    step = suggested_velocity_step("Halpha", Z, LSF)
    assert step == pytest.approx(284.0, rel=0.05)


def test_fixed_ratio_doublet_is_tied_into_one_amplitude():
    tied = system().component_names
    free = system(tie_fixed_ratios=False).component_names
    assert set(tied) == {"Halpha", "NII6584"}
    assert set(free) == {"Halpha", "NII6584", "NII6548"}


def test_density_sensitive_doublet_stays_free():
    s = system(line_names=("Halpha", "SII6716", "SII6731"))
    assert set(s.component_names) == {"Halpha", "SII6716", "SII6731"}


def test_unknown_line_is_rejected():
    with pytest.raises(ValueError):
        system(line_names=("NotALine",))


def test_velocity_basis_columns_are_unit_flux_and_ordered():
    w = sir_wavelength_grid()
    s = system()
    basis = s.velocity_basis(w, "Halpha", SIR_BINWIDTH_ANGSTROM)
    assert basis.shape == (w.size, s.n_velocities)
    integrals = basis.sum(axis=0) * SIR_BINWIDTH_ANGSTROM
    assert np.allclose(integrals, 1.0, rtol=1e-4)
    peaks = w[np.argmax(basis, axis=0)]
    assert np.all(np.diff(peaks) >= 0)


def test_smoothness_penalty_annihilates_a_linear_profile():
    d = smoothness_penalty(9)
    assert np.allclose(d @ np.arange(9, dtype=float), 0.0)
    assert smoothness_penalty(2).size == 0


def test_normalise_profile_gives_unit_integral():
    v = velocity_grid(1000.0, 250.0)
    p = normalise_profile(np.ones(v.size), v)
    assert p.sum() * 250.0 == pytest.approx(1.0)
    with pytest.raises(ValueError):
        normalise_profile(np.zeros(v.size), v)


def make_data(lines, seed=0, noise=2e-19, continuum=1e-17, lsf=LSF, masked=0.0):
    truth = SpectrumTruth(
        z=Z,
        continuum_flux=continuum,
        lines=tuple(lines),
        lsf_sigma=lsf,
        noise_flux=noise,
        seed=seed,
        masked_fraction=masked,
    )
    return simulate_arrays(truth)


def test_recovers_narrow_line_fluxes_from_a_clean_synthetic_spectrum():
    flux_ha, flux_nii = 4.0e-16, 1.2e-16
    w, f, v, m, q = make_data(
        [
            LineTruth("Halpha", flux_ha, 150.0),
            LineTruth("NII6584", flux_nii, 150.0),
            LineTruth("NII6548", flux_nii / 2.94, 150.0),
        ]
    )
    design, _, _ = continuum_block(w, n_knots=6)
    fit = fit_narrow_system(
        w, f, v, system(), continuum_design=design, bin_width=SIR_BINWIDTH_ANGSTROM
    )
    assert fit.amplitude("Halpha") == pytest.approx(flux_ha, rel=0.08)
    assert fit.amplitude("NII6584") == pytest.approx(flux_nii, rel=0.20)
    assert fit.solution.chi2_reduced == pytest.approx(1.0, rel=0.3)


def test_profile_integral_is_one_and_amplitudes_carry_the_flux():
    w, f, v, m, q = make_data([LineTruth("Halpha", 3.0e-16, 200.0)])
    design, _, _ = continuum_block(w, n_knots=6)
    fit = fit_narrow_system(
        w, f, v, system(line_names=("Halpha",)), continuum_design=design,
        bin_width=SIR_BINWIDTH_ANGSTROM,
    )
    step = 250.0
    assert fit.profile.sum() * step == pytest.approx(1.0, rel=1e-6)
    assert fit.amplitude("Halpha") == pytest.approx(3.0e-16, rel=0.1)


def test_amplitudes_are_non_negative_on_a_pure_noise_spectrum():
    w, f, v, m, q = make_data([], seed=7)
    design, _, _ = continuum_block(w, n_knots=6)
    fit = fit_narrow_system(
        w, f, v, system(), continuum_design=design, bin_width=SIR_BINWIDTH_ANGSTROM
    )
    assert np.all(fit.amplitudes >= 0.0)


def test_unpenalised_alternation_is_monotone():
    """With no penalty the objective is fixed, so chi2 can only fall."""
    w, f, v, m, q = make_data(
        [LineTruth("Halpha", 5.0e-16, 300.0), LineTruth("NII6584", 2.0e-16, 300.0)]
    )
    design, _, _ = continuum_block(w, n_knots=6)
    fit = fit_narrow_system(
        w, f, v, system(), continuum_design=design, bin_width=SIR_BINWIDTH_ANGSTROM,
        smoothness=0.0,
    )
    history = np.asarray(fit.chi2_history)
    assert np.all(np.diff(history) <= 1e-6 * np.abs(history[:-1]))
    assert fit.converged


def test_penalised_alternation_converges_without_drifting_up():
    """With a scale-free penalty the objective is re-normalised each step.

    The penalty strength is set relative to the design matrix so that one
    dimensionless number works for bright and faint objects alike, and the
    design changes between half-steps.  The data chi2 is therefore not exactly
    monotone.  It must still converge, and the drift must be negligible: here
    it is bounded at 1 per cent of chi2.
    """
    w, f, v, m, q = make_data(
        [LineTruth("Halpha", 5.0e-16, 300.0), LineTruth("NII6584", 2.0e-16, 300.0)]
    )
    design, _, _ = continuum_block(w, n_knots=6)
    fit = fit_narrow_system(
        w, f, v, system(), continuum_design=design, bin_width=SIR_BINWIDTH_ANGSTROM,
        smoothness=1.0,
    )
    history = np.asarray(fit.chi2_history)
    assert fit.converged
    assert history[-1] <= history[0]
    assert np.max(np.diff(history)) < 0.01 * history[-1]


def test_smoothness_changes_the_recovered_profile():
    w, f, v, m, q = make_data([LineTruth("Halpha", 6.0e-16, 200.0)], seed=13)
    design, _, _ = continuum_block(w, n_knots=6)
    roughness = []
    for smoothness in (0.0, 1.0, 10.0):
        fit = fit_narrow_system(
            w, f, v, system(line_names=("Halpha",)), continuum_design=design,
            bin_width=SIR_BINWIDTH_ANGSTROM, smoothness=smoothness,
        )
        roughness.append(float(np.sum(np.diff(fit.profile, 2) ** 2)))
    assert roughness[0] > roughness[-1]


def test_recovered_profile_width_tracks_the_injected_width():
    """The profile is non-parametric, so this is a weak but real check."""
    widths = []
    for sigma in (150.0, 600.0):
        w, f, v, m, q = make_data([LineTruth("Halpha", 8.0e-16, sigma)], seed=11)
        design, _, _ = continuum_block(w, n_knots=6)
        fit = fit_narrow_system(
            w,
            f,
            v,
            system(line_names=("Halpha",), velocities=velocity_grid(1500.0, 250.0)),
            continuum_design=design,
            bin_width=SIR_BINWIDTH_ANGSTROM,
            smoothness=0.3,
        )
        widths.append(fit.profile_moments()["sigma_kms"])
    assert widths[1] > widths[0]


def test_masked_pixels_are_simply_absent_from_the_fit():
    w, f, v, m, q = make_data([LineTruth("Halpha", 4e-16, 150.0)], masked=0.2, seed=3)
    ok = (m & 1) == 0
    design, _, _ = continuum_block(w[ok], n_knots=6)
    fit = fit_narrow_system(
        w[ok], f[ok], v[ok], system(), continuum_design=design,
        bin_width=SIR_BINWIDTH_ANGSTROM,
    )
    assert fit.solution.n_pixels == int(ok.sum())
    assert fit.amplitude("Halpha") > 0


def test_starting_profile_does_not_change_the_answer_much():
    w, f, v, m, q = make_data([LineTruth("Halpha", 6e-16, 250.0)], seed=5)
    design, _, _ = continuum_block(w, n_knots=6)
    results = []
    for start in (
        gaussian_profile(velocity_grid(1000.0, 250.0), 150.0),
        gaussian_profile(velocity_grid(1000.0, 250.0), 700.0),
        normalise_profile(np.ones(velocity_grid(1000.0, 250.0).size), velocity_grid(1000.0, 250.0)),
    ):
        fit = fit_narrow_system(
            w, f, v, system(line_names=("Halpha",)), continuum_design=design,
            bin_width=SIR_BINWIDTH_ANGSTROM, initial_profile=start,
        )
        results.append(fit.amplitude("Halpha"))
    assert np.std(results) / np.mean(results) < 0.05
