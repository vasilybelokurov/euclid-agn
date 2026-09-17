import numpy as np

from euclid_agn.fit.galaxy_engine import GalaxyEngine
from euclid_agn.fit.quality import ZWarn
from euclid_agn.fit.template_cube import CubeStore, redshift_grid
from euclid_agn.spectra.types import CombinedSpectrum, SourceContext, SpectralObservation
from test_joint_scan import continuum_templates, observe


def as_observation(spectrum):
    combined = CombinedSpectrum(wavelength=spectrum.wavelength, flux=spectrum.flux, variance=spectrum.variance,
                                mask=spectrum.mask, quality=spectrum.quality, lsf_sigma=spectrum.lsf_sigma, bin_width=spectrum.bin_width)
    return SpectralObservation(source=SourceContext(object_id=1, ra=0.0, dec=0.0, tile_id=0), combined=combined, dithers=())


def engine():
    w = observe(0.1).wavelength
    return GalaxyEngine(CubeStore({"GALAXY": continuum_templates()}, redshift_grid(0.0, 2.0, 300.0), w, 13.4))


def test_bright_line_free_galaxy_uses_continuum_model():
    res = engine().run(as_observation(observe(0.1, line_flux=0.0, snr=40.0)))
    assert res.chosen == "continuum" and abs(res.z - 0.1) < 0.004
    assert not (res.zwarn & ZWarn.MODELS_DISAGREE) or res.lines is not None


def test_faint_emission_line_galaxy_uses_line_model():
    # a continuum shape the template set does not contain (steep blue power law), as for real
    # faint ELGs: the continuum model is then inadequate and the lines+spline model must win
    spectrum = observe(1.2, line_flux=1.2e-15, snr=1.5)
    w = spectrum.wavelength
    from euclid_agn.spectra.types import Spectrum1D
    alien = 1e-17 * ((w / 15000.0) ** -2.5 - 0.6) * 0.8
    spectrum = Spectrum1D(wavelength=w, flux=spectrum.flux + alien, variance=spectrum.variance, mask=spectrum.mask,
                          quality=spectrum.quality, lsf_sigma=spectrum.lsf_sigma, bin_width=spectrum.bin_width)
    res = engine().run(as_observation(spectrum), z_prior=1.2)
    # at S/N 1.5 the [N II]/[S II] companions are invisible, so H-alpha vs Pa-beta is a tie the
    # prior must break: assert on the prior-resolved redshift
    assert res.chosen == "lines" and abs(res.result.z_prior - 1.2) < 0.004
    row = res.as_row()
    assert row["gal_model"] == "lines" and np.isfinite(row["gal_bic_continuum"])
