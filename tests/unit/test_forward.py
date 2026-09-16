"""End-to-end synthetic tests of the M0 / M1 hypothesis fit."""

from __future__ import annotations

import numpy as np
import pytest

from euclid_agn.constants import SIR_BINWIDTH_ANGSTROM
from euclid_agn.models.broad import BroadFamily
from euclid_agn.models.forward import assemble, fit_hypothesis
from euclid_agn.models.narrow import NarrowSystem, gaussian_profile, velocity_grid
from euclid_agn.validation.simulator import LineTruth, SpectrumTruth, simulate_arrays

Z = 1.2
LSF = 13.7
VELOCITIES = velocity_grid(1000.0, 250.0)


def narrow_system(**kwargs) -> NarrowSystem:
    base = dict(
        line_names=("Halpha", "NII6584", "NII6548"), z=Z, lsf_sigma=LSF, velocities=VELOCITIES
    )
    base.update(kwargs)
    return NarrowSystem(**base)


def broad_family(sigma_kms=2500.0, **kwargs) -> BroadFamily:
    base = dict(line_names=("Halpha",), z=Z, sigma_kms=sigma_kms, lsf_sigma=LSF)
    base.update(kwargs)
    return BroadFamily(**base)


def make(lines, seed=0, noise=2e-19, continuum=1e-17):
    truth = SpectrumTruth(
        z=Z,
        continuum_flux=continuum,
        lines=tuple(lines),
        lsf_sigma=LSF,
        noise_flux=noise,
        seed=seed,
    )
    w, f, v, m, q = simulate_arrays(truth)
    return w, f, v


NARROW_ONLY = [
    LineTruth("Halpha", 4.0e-16, 150.0),
    LineTruth("NII6584", 1.2e-16, 150.0),
    LineTruth("NII6548", 1.2e-16 / 2.94, 150.0),
]


def test_assemble_labels_and_bounds_every_block():
    w = np.linspace(12500.0, 18500.0, 400)
    blocks = assemble(
        w,
        narrow=narrow_system(),
        narrow_profile=gaussian_profile(VELOCITIES),
        broad=broad_family(),
        bin_width=SIR_BINWIDTH_ANGSTROM,
    )
    assert set(blocks.slices) == {"continuum", "narrow", "broad"}
    assert np.all(np.isneginf(blocks.lower[blocks.slices["continuum"]]))
    assert np.all(blocks.lower[blocks.slices["narrow"]] == 0.0)
    assert np.all(blocks.lower[blocks.slices["broad"]] == 0.0)
    assert blocks.names[blocks.slices["broad"]][0] == "broad_Halpha"
    assert blocks.regularisation.shape[1] == blocks.n_parameters


def test_assemble_requires_a_profile_for_the_narrow_block():
    with pytest.raises(ValueError):
        assemble(np.linspace(12500.0, 18500.0, 100), narrow=narrow_system())


def test_no_false_broad_detection_in_a_noiseless_pure_narrow_spectrum():
    """The headline null test: narrow-only truth must not prefer a BLR."""
    truth = SpectrumTruth(
        z=Z, continuum_flux=1e-17, lines=tuple(NARROW_ONLY), lsf_sigma=LSF, noise_flux=0.0
    )
    from euclid_agn.validation.simulator import noiseless_spectrum

    w, f = noiseless_spectrum(truth)
    variance = np.full(w.size, (1e-19) ** 2)
    fit = fit_hypothesis(
        w, f, variance, narrow_system(), broad_family(), bin_width=SIR_BINWIDTH_ANGSTROM
    )
    assert fit.broad_fluxes["Halpha"] / 4.0e-16 < 0.05
    assert fit.delta_chi2 < 0.05 * fit.m0.chi2 + 25.0


def test_recovers_an_injected_broad_line():
    lines = [*NARROW_ONLY, LineTruth("Halpha", 1.5e-15, 2500.0, broad=True)]
    w, f, v = make(lines, seed=2)
    fit = fit_hypothesis(
        w, f, v, narrow_system(), broad_family(2500.0), bin_width=SIR_BINWIDTH_ANGSTROM
    )
    assert fit.broad_fluxes["Halpha"] == pytest.approx(1.5e-15, rel=0.25)
    assert fit.delta_chi2 > 100.0
    assert fit.m1.chi2 < fit.m0.chi2


def test_broad_flux_stays_at_zero_when_there_is_none():
    w, f, v = make(NARROW_ONLY, seed=4)
    fit = fit_hypothesis(
        w, f, v, narrow_system(), broad_family(2500.0), bin_width=SIR_BINWIDTH_ANGSTROM
    )
    assert fit.broad_fluxes["Halpha"] >= 0.0
    assert fit.delta_chi2 >= 0.0


def test_delta_chi2_increases_with_injected_broad_flux():
    """More signal must not produce less evidence."""
    statistics = []
    for broad_flux in (0.0, 5e-16, 1.5e-15, 4e-15):
        lines = list(NARROW_ONLY)
        if broad_flux > 0:
            lines.append(LineTruth("Halpha", broad_flux, 2500.0, broad=True))
        w, f, v = make(lines, seed=9)
        fit = fit_hypothesis(
            w, f, v, narrow_system(), broad_family(2500.0), bin_width=SIR_BINWIDTH_ANGSTROM
        )
        statistics.append(fit.delta_chi2)
    assert statistics == sorted(statistics)


def test_recovered_broad_flux_increases_with_injected_broad_flux():
    recovered = []
    for broad_flux in (5e-16, 1.5e-15, 4e-15):
        lines = [*NARROW_ONLY, LineTruth("Halpha", broad_flux, 2500.0, broad=True)]
        w, f, v = make(lines, seed=6)
        fit = fit_hypothesis(
            w, f, v, narrow_system(), broad_family(2500.0), bin_width=SIR_BINWIDTH_ANGSTROM
        )
        recovered.append(fit.broad_fluxes["Halpha"])
    assert recovered == sorted(recovered)


def test_narrow_fluxes_survive_the_addition_of_a_broad_component():
    """A BLR must not simply eat the narrow line."""
    lines = [*NARROW_ONLY, LineTruth("Halpha", 1.5e-15, 3000.0, broad=True)]
    w, f, v = make(lines, seed=8)
    fit = fit_hypothesis(
        w, f, v, narrow_system(), broad_family(3000.0), bin_width=SIR_BINWIDTH_ANGSTROM
    )
    assert fit.narrow_fluxes["Halpha"] > 0.3 * 4.0e-16


def test_summary_is_flat_and_carries_the_selection_function_axes():
    lines = [*NARROW_ONLY, LineTruth("Halpha", 1e-15, 2000.0, broad=True)]
    w, f, v = make(lines, seed=1)
    fit = fit_hypothesis(
        w, f, v, narrow_system(), broad_family(2000.0), bin_width=SIR_BINWIDTH_ANGSTROM
    )
    summary = fit.summary()
    for key in (
        "z",
        "lsf_sigma_angstrom",
        "chi2_m0",
        "chi2_m1",
        "delta_chi2",
        "broad_fwhm_kms",
        "broad_resolution_ratio",
        "broad_flux_Halpha",
        "narrow_flux_Halpha",
        "narrow_profile_sigma_kms",
    ):
        assert key in summary, key
    assert all(not isinstance(v, np.ndarray) for v in summary.values())


def test_fit_without_a_broad_family_reports_zero_evidence():
    w, f, v = make(NARROW_ONLY, seed=3)
    fit = fit_hypothesis(w, f, v, narrow_system(), None, bin_width=SIR_BINWIDTH_ANGSTROM)
    assert fit.m1 is None
    assert fit.delta_chi2 == 0.0
    assert fit.broad_fluxes == {}
