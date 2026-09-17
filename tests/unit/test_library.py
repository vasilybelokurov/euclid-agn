"""Template library: loading, telluric handling, projection, PCA."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from euclid_agn.fit.screen import ScreenSettings, prepare
from euclid_agn.models.library import (
    DEFAULT_ROOT, XSL_TELLURIC_GAPS, Template, build_pca_basis, load_glikman_composite,
    load_xsl_ssp, load_xsl_ssp_library, project_template,
)
from euclid_agn.validation.simulator import SpectrumTruth, simulate_arrays
from euclid_agn.spectra.types import Spectrum1D

HAVE_XSL = any(Path(DEFAULT_ROOT).glob("xsl_ssp/**/*.fits"))
HAVE_QSO = (Path(DEFAULT_ROOT) / "qso" / "table7.dat").exists()
needs_xsl = pytest.mark.skipif(not HAVE_XSL, reason="XSL SSP library not downloaded")
needs_qso = pytest.mark.skipif(not HAVE_QSO, reason="Glikman composite not downloaded")


def synthetic_template(gap=True):
    w = np.exp(np.arange(np.log(8000.0), np.log(19000.0), 1e-4))
    flux = 1.0 + 0.3 * np.sin(w / 800.0) + (w / 15000.0) ** -1.5
    mask = np.ones(w.size, bool)
    if gap:
        bad = (w > 13500) & (w < 14250)
        flux = flux.copy(); flux[bad] *= 1 + 0.5 * np.random.default_rng(0).normal(size=bad.sum())
        mask &= ~bad
    return Template("syn", "GALAXY", w, flux, mask)


def test_bridging_replaces_the_gap_smoothly_and_leaves_clean_pixels_alone():
    t = synthetic_template(); b = t.bridged()
    gap = ~t.mask
    assert np.allclose(b.flux[~gap], t.flux[~gap])
    assert np.std(np.diff(b.flux[gap])) < 0.05 * np.std(np.diff(t.flux[gap]))
    assert b.mask.all()
    # a template with no gap is returned unchanged
    clean = synthetic_template(gap=False)
    assert clean.bridged() is clean


def host():
    w, f, v, m, q = simulate_arrays(SpectrumTruth(z=0.3, seed=1))
    return Spectrum1D(wavelength=w, flux=f, variance=v, mask=m, quality=q, lsf_sigma=15.0, bin_width=13.4)


def test_projection_is_lsf_smoothed_pixel_integrated_and_unit_rms():
    t = synthetic_template(gap=False)
    projected = prepare(host(), ScreenSettings())
    column = project_template(t, projected, 0.3)
    assert column is not None and column.shape == projected.wavelength.shape
    assert np.sqrt(np.mean(column**2)) == pytest.approx(1.0)
    # the sine wiggle (period 800 A rest ~ 1040 A observed) survives smoothing by a 15 A LSF
    assert np.std(column) > 0.05
    # out of coverage -> None
    assert project_template(t, projected, 3.0) is None


def test_pca_basis_reconstructs_its_inputs():
    rng = np.random.default_rng(2)
    base = synthetic_template(gap=False)
    templates = [Template(f"t{i}", "GALAXY", base.wavelength, base.flux * (1 + a * (base.wavelength / 12000 - 1)), base.mask)
                 for i, a in enumerate(rng.uniform(-0.5, 0.5, 12))]
    basis = build_pca_basis(templates, n_components=2, wmin=9000, wmax=18000)
    assert basis.n_components == 2
    assert basis.explained[0] > 0.9
    assert len(basis.templates()) == 3


@needs_xsl
def test_xsl_ssp_loads_with_telluric_mask_and_metadata():
    files = sorted(Path(DEFAULT_ROOT).glob("xsl_ssp/**/*.fits"))
    t = load_xsl_ssp(files[0])
    assert t.kind == "GALAXY" and t.wavelength[0] == pytest.approx(3500.0, rel=1e-3)
    assert t.wavelength[-1] > 24000
    for lo, hi in XSL_TELLURIC_GAPS:
        inside = (t.wavelength > lo) & (t.wavelength < min(hi, t.wavelength[-1]))
        assert not t.mask[inside].any()
    assert "log_age" in t.metadata and "mh" in t.metadata


@needs_xsl
def test_xsl_library_selection_and_basis():
    lib = load_xsl_ssp_library(mh_min=-0.5, log_age_min=9.0)
    assert 20 < len(lib) < 200
    assert all(t.metadata["mh"] >= -0.5 for t in lib)
    basis = build_pca_basis(lib, n_components=4)
    assert basis.explained.sum() > 0.95
    projected = prepare(host(), ScreenSettings())
    column = project_template(basis.templates()[0], projected, 0.3)
    assert column is not None and np.all(np.isfinite(column))


@needs_qso
def test_glikman_composite_loads_and_projects():
    q = load_glikman_composite()
    assert q.kind == "QSO" and q.wavelength[0] < 3000 and q.wavelength[-1] > 35000
    projected = prepare(host(), ScreenSettings())
    for z in (0.1, 1.2, 3.5):
        column = project_template(q, projected, z)
        assert column is not None and np.all(np.isfinite(column)), z


HAVE_XSL_DR3 = any(Path(DEFAULT_ROOT).glob("xsl_dr3/**/xsl_spectrum_*_merged.fits"))


@pytest.mark.skipif(not HAVE_XSL_DR3, reason="XSL DR3 not downloaded")
def test_xsl_dr3_star_loads_in_angstrom_with_gaps_masked():
    from euclid_agn.models.library import load_xsl_dr3_star

    f = sorted(Path(DEFAULT_ROOT).glob("xsl_dr3/**/xsl_spectrum_*_merged.fits"))[0]
    t = load_xsl_dr3_star(f)
    assert t.kind == "STAR" and 3400 < t.wavelength[0] < 3600 and 24000 < t.wavelength[-1] < 25000
    assert not t.mask[(t.wavelength > 13600) & (t.wavelength < 14200)].any()
    assert t.mask[(t.wavelength > 15000) & (t.wavelength < 17000)].mean() > 0.95
    assert np.isfinite(t.bridged().flux).all()


def test_bridging_interpolates_isolated_defects_and_cubics_wide_gaps():
    from euclid_agn.models.library import Template

    w = np.exp(np.arange(np.log(9000.0), np.log(20000.0), 1e-4))
    flux = (w / 15000.0) ** -1.0
    mask = np.ones(w.size, bool)
    mask[500] = False; mask[1200:1203] = False  # single-pixel and 3-pixel defects
    gap = (w > 13500) & (w < 14250); mask[gap] = False  # telluric gap
    t = Template("toy", "GALAXY", w, np.where(mask, flux, np.nan), mask)
    b = t.bridged()
    assert np.isfinite(b.flux).all()
    assert np.allclose(b.flux[[500, 1200, 1201, 1202]], flux[[500, 1200, 1201, 1202]], rtol=1e-3)
    assert np.allclose(b.flux[gap], flux[gap], rtol=2e-2)


def test_smoothing_cache_is_per_instance_not_per_id():
    from euclid_agn.models.library import Template, smoothed_cumulative

    w = np.exp(np.arange(np.log(9000.0), np.log(20000.0), 1e-4))
    a = Template("a", "GALAXY", w, np.ones(w.size), np.ones(w.size, bool))
    _, ca = smoothed_cumulative(a, 1e-3)
    b = Template("b", "GALAXY", w, 2 * np.ones(w.size), np.ones(w.size, bool))
    _, cb = smoothed_cumulative(b, 1e-3)
    assert np.allclose(cb, 2 * ca)
    assert "_smoothed_cache" in a.__dict__ and "_smoothed_cache" in b.__dict__


HAVE_PHOENIX = any(Path(DEFAULT_ROOT).glob("phoenix/Z*/lte*.fits"))


@pytest.mark.skipif(not HAVE_PHOENIX, reason="PHOENIX R10000 grid not downloaded")
def test_phoenix_star_loads_on_a_log_grid_without_gaps():
    from euclid_agn.models.library import load_phoenix_library, load_phoenix_star

    f = sorted(Path(DEFAULT_ROOT).glob("phoenix/Z*/lte05500-4.50*.fits"))[0]
    t = load_phoenix_star(f)
    assert t.kind == "STAR" and abs(t.wavelength[0] - 3000.0) < 0.5 and 24900 < t.wavelength[-1] < 25100
    assert np.allclose(np.diff(np.log(t.wavelength)), 1e-5, rtol=1e-6)
    assert t.mask.all() and t.metadata["teff"] == 5500.0 and t.metadata["logg"] == 4.5
    lib = load_phoenix_library(teff_min=5000, teff_max=5200, mh_values=(0.0,))
    assert all(5000 <= x.metadata["teff"] <= 5200 for x in lib) and len(lib) >= 5
