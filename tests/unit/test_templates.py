"""Fixed-ratio templates must give evidence *against* wrong identifications."""

from __future__ import annotations

import numpy as np
import pytest

from euclid_agn.constants import C_KMS
from euclid_agn.fit.hypotheses import RedshiftHypothesis, blind_grid
from euclid_agn.fit.screen import ScreenSettings, matched_filter, prepare, quick_scan
from euclid_agn.models.line_catalog import BY_NAME, SYSTEMS
from euclid_agn.models.templates import TEMPLATES, template_column, templates_for
from euclid_agn.spectra.types import Spectrum1D
from euclid_agn.validation.simulator import LineTruth, SpectrumTruth, simulate_arrays

SETTINGS = ScreenSettings(broad_sigma_kms=(600.0, 1500.0), n_refine=0, n_local=3)


def make(lines, z, seed=0, noise=2e-19):
    truth = SpectrumTruth(z=z, continuum_flux=1e-17, lines=tuple(lines), lsf_sigma=13.7,
                          noise_flux=noise, seed=seed)
    w, f, v, m, q = simulate_arrays(truth)
    return Spectrum1D(wavelength=w, flux=f, variance=v, mask=m, quality=q, lsf_sigma=13.7,
                      bin_width=13.4)


def test_every_template_references_known_lines_and_a_known_system():
    systems = {s.name for s in SYSTEMS}
    for template in TEMPLATES:
        assert template.system in systems
        for name in template.ratios:
            assert name in BY_NAME, name
        assert all(r > 0 for r in template.ratios.values())


def test_every_system_has_at_least_one_template():
    for system in SYSTEMS:
        assert templates_for(system), system.name


def test_template_column_is_the_ratio_weighted_sum_of_unit_lines():
    projected = prepare(make([], 1.2), SETTINGS)
    template = templates_for("halpha_complex")[0]
    column, n = template_column(projected, template, 1.2, 200.0)
    assert n == 6
    # total flux equals the sum of the ratios (each line has unit flux)
    assert np.sum(column * projected.widths) == pytest.approx(sum(template.ratios.values()), rel=1e-3)


def test_template_penalises_a_wrong_identification_the_free_fit_cannot():
    """H-beta + [O III] at z=1.965 vs H-alpha complex at z=1.2: 1.6 pixels apart.

    With free amplitudes the wrong system scores about as well as the truth.
    With fixed ratios the H-alpha template must put [N II] where the spectrum
    has nothing, and loses clearly.
    """
    z_true = 1.965
    spectrum = make(
        [LineTruth("Hbeta", 3e-16, 150.0), LineTruth("OIII5007", 1.2e-15, 150.0),
         LineTruth("OIII4959", 4e-16, 150.0)],
        z_true, seed=3,
    )
    z_wrong = BY_NAME["Hbeta"].rest * (1 + z_true) / BY_NAME["Halpha"].rest - 1.0
    hypotheses = [
        RedshiftHypothesis(z_true, "test", "hbeta_oiii"),
        RedshiftHypothesis(z_wrong, "test", "halpha_complex"),
    ]
    scan = quick_scan(spectrum, hypotheses, SETTINGS).set_index("system")
    free_ratio = scan.loc["halpha_complex", "delta_chi2_total"] / scan.loc["hbeta_oiii", "delta_chi2_total"]
    template_ratio = (
        scan.loc["halpha_complex", "delta_chi2_identification"]
        / scan.loc["hbeta_oiii", "delta_chi2_identification"]
    )
    assert template_ratio < free_ratio
    assert template_ratio < 0.7
    assert scan["rank_statistic"].idxmax() == "hbeta_oiii"


def test_blind_scan_with_templates_recovers_an_hii_like_redshift():
    z_true = 1.30
    spectrum = make(
        [LineTruth("Halpha", 5e-16, 150.0), LineTruth("NII6584", 1.5e-16, 150.0),
         LineTruth("NII6548", 5e-17, 150.0), LineTruth("SII6716", 1e-16, 150.0),
         LineTruth("SII6731", 7e-17, 150.0)],
        z_true, seed=5,
    )
    scan = quick_scan(spectrum, blind_grid(step_kms=600.0), SETTINGS)
    best = scan.loc[scan["rank_statistic"].idxmax()]
    assert best["system"] == "halpha_complex"
    assert abs(C_KMS * (best["z"] - z_true) / (1 + z_true)) < 700.0
    assert best["template"].startswith("halpha")


def test_rank_by_can_fall_back_to_the_penalised_statistic():
    spectrum = make([LineTruth("Halpha", 5e-16, 150.0)], 1.2, seed=6)
    hyp = [RedshiftHypothesis(1.2, "test", "halpha_complex")]
    a = quick_scan(spectrum, hyp, ScreenSettings(n_refine=0, rank_by="template"))
    b = quick_scan(spectrum, hyp, ScreenSettings(n_refine=0, rank_by="penalised"))
    assert a["rank_statistic"].iloc[0] == pytest.approx(a["delta_chi2_identification"].iloc[0])
    assert b["rank_statistic"].iloc[0] == pytest.approx(b["delta_chi2_penalised"].iloc[0])


def test_template_lines_on_the_edge_are_dropped():
    """The Pa-gamma-on-the-red-edge failure from real data."""
    projected = prepare(make([], 0.688), SETTINGS)
    template = [t for t in templates_for("helium_paschen_gamma") if t.name == "paschen_only"][0]
    # At z=0.688 Pa-gamma sits at 18468 A, within 3 pixels of the edge.
    strict, n_strict = template_column(projected, template, 0.688, 200.0, min_containment=0.8)
    loose, n_loose = template_column(projected, template, 0.688, 200.0, min_containment=0.0)
    assert n_loose > n_strict
    edge = projected.wavelength > 18400
    assert np.sum(strict[edge]) < 0.1 * np.sum(loose[edge])
