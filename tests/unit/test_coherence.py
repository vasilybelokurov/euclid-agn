import numpy as np

from euclid_agn.spectra.coherence import apply_coherence_mask, dither_coherence
from euclid_agn.spectra.types import CombinedSpectrum, DitherSpectrum, SourceContext, SpectralObservation
from euclid_agn.validation.simulator import sir_wavelength_grid


def make_observation(seed=0, contaminate=True):
    w = sir_wavelength_grid(); n = w.size
    rng = np.random.default_rng(seed)
    truth = 1e-17 * (w / 15000.0) ** -1.0
    sigma = 1e-18
    dithers = []
    for i in range(4):
        f = truth + rng.normal(0, sigma, n)
        mask = np.zeros(n, int)
        if contaminate and i == 2:
            f[300:340] += 8e-18  # one dither carries a contaminating order
        if i == 0:
            mask[400:420] = 1  # one dither masked here: 3 usable remain
        dithers.append(DitherSpectrum(wavelength=w, flux=f, variance=np.full(n, sigma**2), mask=mask, quality=np.ones(n),
                                      lsf_sigma=14.0, bin_width=13.4, dither_id=i, pointing_id=i))
    stack = np.mean([d.flux for d in dithers], axis=0)
    combined = CombinedSpectrum(wavelength=w, flux=stack, variance=np.full(n, sigma**2 / 4), mask=np.zeros(n, int),
                                quality=np.ones(n), lsf_sigma=14.0, bin_width=13.4)
    return SpectralObservation(source=SourceContext(object_id=1, ra=0.0, dec=0.0, tile_id=0), combined=combined, dithers=tuple(dithers))


def test_incoherent_pixels_are_flagged_and_coherent_ones_are_not():
    obs = make_observation()
    rep = dither_coherence(obs, grow=1)
    assert 0.5 < rep.median_reduced_chi2 < 2.0
    assert rep.bad[300:340].all()
    assert rep.bad[299] and rep.bad[340]  # grown by one pixel
    assert not rep.bad[:250].any() and not rep.bad[360:].any()
    assert rep.n_usable[405] == 3


def test_clean_observation_flags_almost_nothing_and_mask_is_applied():
    obs = make_observation(contaminate=False)
    rep = dither_coherence(obs)
    assert rep.n_bad <= 3
    spectrum, rep2 = apply_coherence_mask(make_observation())
    assert spectrum.metadata["extra_mask_coherence"] == rep2.n_bad
    assert not spectrum.usable()[300:340].any()
    assert obs.combined.mask.sum() == 0  # original untouched


def test_variance_rescale_tracks_local_dither_scatter():
    from euclid_agn.spectra.coherence import dither_variance_rescale

    obs = make_observation(contaminate=True)
    spectrum, rep = dither_variance_rescale(obs)
    ratio = spectrum.variance / obs.combined.variance
    assert ratio.min() >= 1.0
    assert np.median(ratio[:250]) < 2.0  # clean region: about the nominal noise
    assert ratio[300:340].mean() > 5.0  # contaminated region: strongly down-weighted
    assert not spectrum.usable()[310:330].any()  # gross contamination still hard-masked
    assert spectrum.metadata["dither_variance_scale_median"] >= 1.0
