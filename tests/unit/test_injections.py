"""Injection/recovery must recover what it injects and find nothing in nulls."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from euclid_agn.fit.screen import ScreenSettings
from euclid_agn.spectra.types import Spectrum1D
from euclid_agn.validation.injections import (
    Injection,
    completeness_table,
    inject_broad_line,
    injection_recovery,
    measurement_row,
    null_forbidden,
    null_off_redshift,
    threshold_for_false_positive_rate,
)
from euclid_agn.validation.simulator import LineTruth, SpectrumTruth, simulate_arrays

Z = 1.2
SETTINGS = ScreenSettings(broad_sigma_kms=(1500.0,), n_refine=0, n_local=0)
NARROW = (
    LineTruth("Halpha", 4e-16, 150.0),
    LineTruth("NII6584", 1.2e-16, 150.0),
    LineTruth("SII6716", 8e-17, 150.0),
    LineTruth("SII6731", 6e-17, 150.0),
)


def host(seed=0, noise=2e-19):
    truth = SpectrumTruth(z=Z, continuum_flux=1e-17, lines=NARROW, lsf_sigma=13.7, noise_flux=noise, seed=seed)
    w, f, v, m, q = simulate_arrays(truth)
    return Spectrum1D(wavelength=w, flux=f, variance=v, mask=m, quality=q, lsf_sigma=13.7, bin_width=13.4)


def test_injection_adds_exactly_the_requested_flux():
    s = host()
    injected = inject_broad_line(s, Z, Injection("Halpha", 2e-15, 2000.0))
    added = (injected.flux - s.flux) * s.bin_width
    assert added.sum() == pytest.approx(2e-15, rel=1e-3)
    assert np.all(injected.variance == s.variance)
    assert injected.metadata["injected"]["flux"] == 2e-15
    assert np.all(s.flux == host().flux)  # the original is untouched


def test_measurement_recovers_an_injected_broad_line():
    s = inject_broad_line(host(seed=1), Z, Injection("Halpha", 2e-15, 1500.0))
    row = measurement_row(s, Z, "halpha_complex", 1500.0, SETTINGS)
    assert row["recovered_flux"] == pytest.approx(2e-15, rel=0.3)
    assert row["delta_chi2"] > 50.0
    assert row["broad_line"] == "Halpha"


def test_on_redshift_null_is_small_without_injection():
    row = measurement_row(host(seed=2), Z, "halpha_complex", 1500.0, SETTINGS)
    assert row["delta_chi2"] < 25.0


def test_completeness_rises_with_injected_flux():
    targets = [(k, host(seed=10 + k), Z, "halpha_complex", 1.0) for k in range(3)]
    table = injection_recovery(targets, fluxes=[2e-17, 3e-16, 4e-15], sigmas_kms=[1500.0], settings=SETTINGS)
    assert set(table["kind"]) == {"on_redshift_null", "injection"}
    comp = completeness_table(table, threshold=25.0).sort_values("injected_flux")
    assert comp["completeness"].iloc[0] <= comp["completeness"].iloc[-1]
    assert comp["completeness"].iloc[-1] == 1.0
    assert comp["flux_ratio_if_detected"].iloc[-1] == pytest.approx(1.0, rel=0.3)


def test_off_redshift_null_finds_no_line():
    targets = [(0, host(seed=3), Z, "halpha_complex", 1.0)]
    null = null_off_redshift(targets, sigmas_kms=[1500.0], settings=SETTINGS)
    # at dz/(1+z) = -0.25 H-alpha leaves the grism, so that offset yields no row
    assert 2 <= len(null) <= 4
    assert (null["delta_chi2"] < 25.0).all()
    assert (null["kind"] == "off_redshift_null").all()


def test_forbidden_null_finds_no_broad_nii():
    targets = [(0, host(seed=4), Z, "halpha_complex", 1.0)]
    null = null_forbidden(targets, sigmas_kms=[1500.0], settings=SETTINGS, forbidden_line="NII6584")
    assert len(null) == 1
    assert null["delta_chi2"].iloc[0] < 25.0
    assert null["broad_line"].iloc[0] == "NII6584"


def test_threshold_from_null_quantile():
    null = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    threshold = threshold_for_false_positive_rate(null, 0.2)
    assert np.mean(null > threshold) <= 0.2
    assert threshold == pytest.approx(100.0)  # "higher": exceeded by at most 20%
    assert np.isnan(threshold_for_false_positive_rate(np.array([]), 0.1))


def test_completeness_table_of_nothing_is_empty():
    assert completeness_table(pd.DataFrame({"kind": []}), 25.0).empty
