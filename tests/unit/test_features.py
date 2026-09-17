import numpy as np
import pytest

from euclid_agn.fit.quality import ZWarn, zwarn_for_row
from euclid_agn.spectra.features import dither_excess, feature_report, strongest_line_feature
from euclid_agn.fit.screen import ScreenSettings, prepare
from euclid_agn.validation.simulator import LineTruth, SpectrumTruth, simulate_observation
import pandas as pd


def test_strongest_feature_finds_an_injected_line():
    obs = simulate_observation(SpectrumTruth(z=1.2, lines=(LineTruth("Halpha", 6e-16, 150.0),), seed=1, noise_flux=5e-19))
    stat, w = strongest_line_feature(prepare(obs.combined, ScreenSettings()))
    assert stat > 50
    assert abs(w - 6564.61 * 2.2) < 20


def test_no_feature_in_a_featureless_spectrum():
    obs = simulate_observation(SpectrumTruth(z=1.2, lines=(), seed=2))
    stat, w = strongest_line_feature(prepare(obs.combined, ScreenSettings()))
    assert stat < 25


def test_a_real_line_is_coherent_across_dithers():
    obs = simulate_observation(SpectrumTruth(z=1.2, lines=(LineTruth("Halpha", 8e-16, 150.0),), seed=3, n_dithers=4))
    report = feature_report(obs)
    assert report.n_dithers_evaluable == 4
    assert report.n_dithers_detected >= 3
    assert report.coherent


def test_a_single_dither_spike_is_incoherent():
    """A neighbour's line in one dither: strong in the combined spectrum, absent elsewhere."""
    from dataclasses import replace

    obs = simulate_observation(SpectrumTruth(z=1.2, lines=(), seed=4, n_dithers=4))
    index = 250
    spike = 40.0 * np.sqrt(obs.dithers[0].variance[index])
    dithers = list(obs.dithers)
    flux0 = dithers[0].flux.copy(); flux0[index - 1 : index + 2] += spike * np.array([0.6, 1.0, 0.6])
    dithers[0] = replace(dithers[0], flux=flux0)
    combined_flux = obs.combined.flux.copy(); combined_flux[index - 1 : index + 2] += spike / 4 * np.array([0.6, 1.0, 0.6])
    obs = replace(obs, combined=replace(obs.combined, flux=combined_flux), dithers=tuple(dithers))
    report = feature_report(obs)
    assert abs(report.wavelength - obs.combined.wavelength[index]) < 20
    assert report.n_dithers_detected == 1
    assert not report.coherent


def test_dither_excess_handles_masked_regions():
    obs = simulate_observation(SpectrumTruth(z=1.2, lines=(), seed=5))
    d = obs.dithers[0]
    from dataclasses import replace
    mask = d.mask.copy(); mask[200:230] = 1
    blocked = replace(d, mask=mask)
    assert dither_excess(blocked, d.wavelength[215]) is None
    assert dither_excess(d, d.wavelength[215]) is not None


def test_quality_bits_for_the_feature_gate():
    row = pd.Series({"feature_max_dchi2": 5.0, "feature_coherent": 0.0, "feature_n_dithers_evaluable": 4, "z_prior": 1.0})
    warn = zwarn_for_row(row)
    assert warn & ZWarn.NO_FEATURE and warn & ZWarn.INCOHERENT_DITHERS
    good = pd.Series({"feature_max_dchi2": 80.0, "feature_coherent": 1.0, "feature_n_dithers_evaluable": 4, "z_prior": 1.0})
    assert zwarn_for_row(good) == 0
