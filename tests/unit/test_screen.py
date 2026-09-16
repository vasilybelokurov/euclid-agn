"""Stage-1 screening: the matched filter must agree with the full solve."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from euclid_agn.constants import C_KMS
from euclid_agn.fit.hypotheses import RedshiftHypothesis
from euclid_agn.fit.screen import (
    ScreenSettings,
    broad_column,
    matched_filter,
    narrow_columns,
    prepare,
    quick_scan,
    refine,
    screen_spectrum,
)
from euclid_agn.models.line_catalog import BY_SYSTEM
from euclid_agn.spectra.types import Spectrum1D
from euclid_agn.validation.simulator import (
    LineTruth,
    SpectrumTruth,
    simulate_arrays,
    sir_wavelength_grid,
)

Z = 1.20
LSF = 13.7
SETTINGS = ScreenSettings(broad_sigma_kms=(500.0, 1500.0, 3000.0), n_refine=1)

NARROW = [
    LineTruth("Halpha", 4.0e-16, 150.0),
    LineTruth("NII6584", 1.2e-16, 150.0),
]


def make(lines, seed=0, noise=2e-19, continuum=1e-17, mask=None, lsf=LSF):
    truth = SpectrumTruth(
        z=Z, continuum_flux=continuum, lines=tuple(lines), lsf_sigma=lsf,
        noise_flux=noise, seed=seed,
    )
    w, f, v, m, q = simulate_arrays(truth)
    if mask is not None:
        m = mask
    return Spectrum1D(
        wavelength=w, flux=f, variance=v, mask=m, quality=q, lsf_sigma=lsf, bin_width=13.4
    )


def test_prepare_orthonormalises_the_continuum_and_removes_it():
    projected = prepare(make(NARROW), SETTINGS)
    assert projected is not None
    assert np.allclose(projected.basis.T @ projected.basis, np.eye(projected.basis.shape[1]), atol=1e-10)
    # The continuum span is gone from the residual.
    assert np.allclose(projected.basis.T @ projected.residual, 0.0, atol=1e-8)
    assert projected.wavelength[0] >= SETTINGS.wavelength_min


def test_prepare_refuses_a_spectrum_with_too_few_usable_pixels():
    n = sir_wavelength_grid().size
    mask = np.ones(n, dtype=int)
    mask[:30] = 0
    assert prepare(make(NARROW, mask=mask), SETTINGS) is None


def test_projection_removes_the_continuum_component_of_a_trial_column():
    projected = prepare(make(NARROW), SETTINGS)
    # A constant is almost entirely continuum; its projection must be tiny.
    constant = np.ones(projected.wavelength.size)
    perpendicular = projected.project(constant)
    whitened_norm = np.linalg.norm(constant * projected.weight)
    assert np.linalg.norm(perpendicular) < 0.05 * whitened_norm


def test_matched_filter_reproduces_the_closed_form_for_one_column():
    projected = prepare(make(NARROW), SETTINGS)
    column = narrow_columns(projected, BY_SYSTEM["halpha_complex"], Z, 150.0)[0][:, :1]
    statistic, amplitudes = matched_filter(projected, column)
    b = projected.project(column)[:, 0]
    expected = (b @ projected.residual) ** 2 / (b @ b)
    assert statistic == pytest.approx(expected, rel=1e-8)
    assert amplitudes[0] == pytest.approx((b @ projected.residual) / (b @ b), rel=1e-8)


def test_matched_filter_returns_zero_for_a_column_that_wants_negative_flux():
    projected = prepare(make([]), SETTINGS)
    column = -np.abs(projected.residual / projected.weight)[:, None]
    statistic, amplitudes = matched_filter(projected, column)
    assert statistic == pytest.approx(0.0, abs=1e-8) or amplitudes[0] >= 0.0


def test_quick_scan_peaks_at_the_true_redshift():
    spectrum = make([*NARROW, LineTruth("Halpha", 2.0e-15, 2500.0, broad=True)], seed=5)
    hypotheses = [
        RedshiftHypothesis(z, "blind", "halpha_complex")
        for z in np.linspace(Z - 0.05, Z + 0.05, 41)
    ]
    scan = quick_scan(spectrum, hypotheses, SETTINGS)
    best = scan.loc[scan["delta_chi2_broad"].idxmax()]
    assert C_KMS * abs(best["z"] - Z) / (1 + Z) < 600.0
    assert best["delta_chi2_broad"] > 50.0


def test_quick_scan_finds_no_broad_signal_in_a_narrow_only_spectrum():
    spectrum = make(NARROW, seed=6)
    hypotheses = [RedshiftHypothesis(Z, "spe_galaxy", "halpha_complex")]
    scan = quick_scan(spectrum, hypotheses, SETTINGS)
    assert scan["delta_chi2_narrow"].iloc[0] > 100.0
    assert scan["delta_chi2_broad"].iloc[0] < 25.0


def test_quick_scan_broad_statistic_grows_with_injected_flux():
    statistics = []
    for flux in (0.0, 5e-16, 2e-15, 6e-15):
        lines = list(NARROW)
        if flux:
            lines.append(LineTruth("Halpha", flux, 2500.0, broad=True))
        spectrum = make(lines, seed=7)
        scan = quick_scan(
            spectrum, [RedshiftHypothesis(Z, "blind", "halpha_complex")], SETTINGS
        )
        statistics.append(float(scan["delta_chi2_broad"].iloc[0]))
    assert statistics == sorted(statistics)


def test_broad_column_is_none_when_the_line_is_off_the_grid():
    projected = prepare(make(NARROW), SETTINGS)
    assert broad_column(projected, "Halpha", 3.0, 2000.0) is None
    assert broad_column(projected, "Halpha", Z, 2000.0) is not None


def test_refine_recovers_the_injected_broad_flux():
    spectrum = make([*NARROW, LineTruth("Halpha", 2.0e-15, 2500.0, broad=True)], seed=8)
    fit = refine(spectrum, Z, "halpha_complex", 2500.0, settings=SETTINGS)
    assert fit is not None
    assert fit.broad_fluxes["Halpha"] == pytest.approx(2.0e-15, rel=0.3)
    assert fit.delta_chi2 > 50.0


def test_screen_spectrum_returns_a_flat_row_with_provenance_and_quality():
    spectrum = make([*NARROW, LineTruth("Halpha", 2.0e-15, 2500.0, broad=True)], seed=9)
    hypotheses = [RedshiftHypothesis(Z, "spe_galaxy", "halpha_complex")]
    table = screen_spectrum(
        spectrum, hypotheses, SETTINGS, object_id=42, noise_inflation=1.41
    )
    assert len(table) == 1
    row = table.iloc[0]
    assert row["object_id"] == 42
    assert row["z_hypothesis_source"] == "spe_galaxy"
    assert row["noise_inflation"] == pytest.approx(1.41)
    assert row["delta_chi2_effective"] == pytest.approx(
        row["quick_delta_chi2_broad"] / 1.41**2
    )
    assert row["delta_chi2_refined_effective"] < row["delta_chi2"]
    assert "quality_usable_pixel_fraction" in row
    assert row["broad_flux_Halpha"] > 0


def test_screen_spectrum_is_empty_for_an_unusable_spectrum():
    n = sir_wavelength_grid().size
    mask = np.ones(n, dtype=int)
    spectrum = make(NARROW, mask=mask)
    assert screen_spectrum(spectrum, [RedshiftHypothesis(Z, "blind")], SETTINGS).empty


def test_noise_inflation_only_ever_reduces_the_statistic():
    spectrum = make([*NARROW, LineTruth("Halpha", 2.0e-15, 2500.0, broad=True)], seed=10)
    hypotheses = [RedshiftHypothesis(Z, "blind", "halpha_complex")]
    plain = screen_spectrum(spectrum, hypotheses, SETTINGS, noise_inflation=1.0)
    inflated = screen_spectrum(spectrum, hypotheses, SETTINGS, noise_inflation=1.41)
    assert inflated["delta_chi2_effective"].iloc[0] < plain["delta_chi2_effective"].iloc[0]
    assert inflated["delta_chi2"].iloc[0] == pytest.approx(plain["delta_chi2"].iloc[0])


def test_halpha_and_pabeta_are_degenerate_to_about_one_pixel():
    """The identification ambiguity that broke the first ranking rule."""
    from euclid_agn.models.line_catalog import BY_NAME

    halpha = BY_NAME["Halpha"].rest * (1 + 1.2)
    z_pabeta = halpha / BY_NAME["Pabeta"].rest - 1.0
    pabeta = BY_NAME["Pabeta"].rest * (1 + z_pabeta)
    assert z_pabeta == pytest.approx(0.1264, abs=0.001)
    assert abs(halpha - pabeta) < 0.01
    # A shift of 0.0012 in Pa-beta redshift is barely more than one pixel.
    offset = BY_NAME["Pabeta"].rest * (1 + 0.1276) - halpha
    assert abs(offset) / 13.4 < 1.5


def test_ranking_by_broad_gain_alone_picks_the_wrong_redshift():
    """Documents why the selection rule uses total evidence."""
    from euclid_agn.fit.hypotheses import blind_grid

    spectrum = make([*NARROW, LineTruth("Halpha", 2.0e-15, 2500.0, broad=True)], seed=0)
    scan = quick_scan(spectrum, blind_grid(0.0, 5.7, step_kms=500.0), SETTINGS)
    scan = scan.assign(delta_chi2_total=scan.delta_chi2_narrow + scan.delta_chi2_broad)
    top_broad = scan.nlargest(5, "delta_chi2_broad")
    top_total = scan.nlargest(5, "delta_chi2_total")
    assert not np.any(np.abs(top_broad["z"] - Z) < 0.01), "broad-only ranking should fail here"
    assert np.any(np.abs(top_total["z"] - Z) < 0.01), "total ranking should find z=1.2"
    assert top_total.iloc[0]["system"] == "halpha_complex"


def test_selection_keeps_both_rankings():
    from euclid_agn.fit.screen import select_for_refinement

    scan = pd.DataFrame(
        {
            "z": [1.2, 0.13, 4.2],
            "system": ["halpha_complex", "paschen_beta", "mgii"],
            # H-alpha wins on total evidence (six lines); Pa-beta wins on the
            # broad gain alone. Both must survive selection.
            "delta_chi2_narrow": [400.0, 200.0, 10.0],
            "delta_chi2_broad": [200.0, 340.0, 20.0],
            "n_components": [7, 3, 3],
        }
    )
    scan["delta_chi2_total"] = scan.delta_chi2_narrow + scan.delta_chi2_broad
    scan["delta_chi2_penalised"] = scan.delta_chi2_total - 6.6 * scan.n_components
    selected = select_for_refinement(scan, n_refine=1)
    assert set(selected["system"]) == {"halpha_complex", "paschen_beta"}
    assert set(selected["selected_by"]) == {"total", "broad"}


def test_ambiguity_is_reported():
    from euclid_agn.fit.screen import rank_alternatives

    scan = pd.DataFrame(
        {
            "z": [1.2, 0.13],
            "system": ["halpha_complex", "paschen_beta"],
            "delta_chi2_total": [500.0, 480.0],
            "delta_chi2_penalised": [500.0, 480.0],
        }
    )
    info = rank_alternatives(scan, scan.iloc[0])
    assert info["delta_chi2_over_other_system"] == pytest.approx(20.0)
    assert info["best_alternative_system"] == "paschen_beta"
    assert info["velocity_to_best_alternative_kms"] > 1e5


def test_screen_reports_the_true_redshift_for_a_broad_line_object():
    """A type-1 host with its full narrow complex must be identified correctly.

    With only H-alpha and [N II] injected the answer is genuinely ambiguous: a
    lone broad feature plus two weak narrow lines is as well explained by Mg II
    at z = 4.16, and after the per-component penalty the smaller system wins.
    That ambiguity is a property of the grism, so the test injects the narrow
    complex a real H-alpha emitter would show.
    """
    from euclid_agn.fit.hypotheses import blind_grid

    full_narrow = [
        *NARROW,
        LineTruth("NII6548", 4.0e-17, 150.0),
        LineTruth("SII6716", 8.0e-17, 150.0),
        LineTruth("SII6731", 6.0e-17, 150.0),
    ]
    spectrum = make([*full_narrow, LineTruth("Halpha", 2.0e-15, 2500.0, broad=True)], seed=0)
    table = screen_spectrum(
        spectrum,
        blind_grid(0.0, 5.7, step_kms=500.0),
        ScreenSettings(broad_sigma_kms=(500.0, 1500.0, 3000.0), n_refine=2),
        object_id=7,
    )
    assert not table.empty
    best = table.iloc[0]
    assert abs(best["z"] - Z) < 0.01
    assert best["system"] == "halpha_complex"
    assert best["broad_flux_Halpha"] > 0
    assert "delta_chi2_over_other_system" in table.columns


def test_continuum_orthogonality_falls_with_line_width():
    """A wide enough Gaussian is indistinguishable from continuum curvature."""
    from euclid_agn.fit.screen import broad_column, continuum_orthogonality

    projected = prepare(make([]), ScreenSettings(n_knots=12))
    values = []
    for sigma in (300.0, 1200.0, 5000.0):
        column = broad_column(projected, "Halpha", Z, sigma)
        values.append(continuum_orthogonality(projected, column))
    assert values == sorted(values, reverse=True)
    assert values[0] > 0.85
    assert values[-1] < 0.25


def test_continuum_orthogonality_falls_with_continuum_flexibility():
    from euclid_agn.fit.screen import broad_column, continuum_orthogonality

    values = []
    for n_knots in (6, 12, 25, 50):
        projected = prepare(make([]), ScreenSettings(n_knots=n_knots))
        column = broad_column(projected, "Halpha", Z, 1200.0)
        values.append(continuum_orthogonality(projected, column))
    assert values == sorted(values, reverse=True)


def test_unidentifiable_widths_are_excluded_from_the_scan():
    from euclid_agn.fit.screen import identifiable_sigmas

    projected = prepare(make([]), ScreenSettings(n_knots=12))
    settings = ScreenSettings(
        n_knots=12, broad_sigma_kms=(300.0, 1200.0, 5000.0, 8000.0),
        min_broad_orthogonality=0.5,
    )
    allowed = identifiable_sigmas(projected, "Halpha", Z, settings)
    widths = [s for s, _ in allowed]
    assert 300.0 in widths and 1200.0 in widths
    assert 5000.0 not in widths and 8000.0 not in widths
    assert all(o >= 0.5 for _, o in allowed)


def test_orthogonality_threshold_suppresses_the_railing_failure():
    """Without the cut, a featureless spectrum prefers the widest width."""
    spectrum = make([], seed=21)
    hypotheses = [RedshiftHypothesis(Z, "blind", "halpha_complex")]
    wide = ScreenSettings(
        n_knots=12, broad_sigma_kms=(500.0, 2500.0, 5000.0), min_broad_orthogonality=0.0
    )
    guarded = ScreenSettings(
        n_knots=12, broad_sigma_kms=(500.0, 2500.0, 5000.0), min_broad_orthogonality=0.5
    )
    unguarded_scan = quick_scan(spectrum, hypotheses, wide)
    guarded_scan = quick_scan(spectrum, hypotheses, guarded)
    assert guarded_scan["broad_sigma_max_identifiable_kms"].iloc[0] <= 2500.0
    kept = guarded_scan["broad_sigma_kms"].iloc[0]
    assert not np.isfinite(kept) or kept <= 2500.0
    assert guarded_scan["delta_chi2_broad"].iloc[0] <= unguarded_scan["delta_chi2_broad"].iloc[0]


def test_orthogonality_is_recorded_on_every_screening_row():
    spectrum = make([*NARROW, LineTruth("Halpha", 2.0e-15, 1500.0, broad=True)], seed=22)
    table = screen_spectrum(
        spectrum, [RedshiftHypothesis(Z, "blind", "halpha_complex")], SETTINGS, object_id=3
    )
    assert "broad_continuum_orthogonality" in table.columns
    assert table["broad_continuum_orthogonality"].iloc[0] >= SETTINGS.min_broad_orthogonality


def test_edge_broad_lines_are_excluded_by_the_containment_rule():
    from euclid_agn.fit.screen import broad_column

    projected = prepare(make([]), ScreenSettings())
    z_edge = projected.wavelength[-1] / 6564.61 - 1.0
    assert broad_column(projected, "Halpha", z_edge, 2000.0, min_containment=0.0) is not None
    assert broad_column(projected, "Halpha", z_edge, 2000.0, min_containment=0.8) is None
    assert broad_column(projected, "Halpha", Z, 2000.0, min_containment=0.8) is not None


def test_screen_records_the_continuum_goodness_of_fit():
    spectrum = make([*NARROW, LineTruth("Halpha", 2.0e-15, 1500.0, broad=True)], seed=31)
    table = screen_spectrum(
        spectrum, [RedshiftHypothesis(Z, "blind", "halpha_complex")], SETTINGS, object_id=5
    )
    assert "chi2_reduced_m0" in table.columns
    assert "continuum_model_ok" in table.columns
    # A correct model on simulated data must pass its own goodness-of-fit test.
    assert table["chi2_reduced_m0"].iloc[0] < 4.0
    assert bool(table["continuum_model_ok"].iloc[0])
