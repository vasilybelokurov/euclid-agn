import numpy as np
import pytest

from euclid_agn.constants import C_KMS
from euclid_agn.fit.hypotheses import (
    RedshiftHypothesis,
    blind_grid,
    deduplicate,
    find_peaks,
    from_catalogue,
    from_peaks,
    generate,
)
from euclid_agn.models.line_catalog import BY_NAME
from euclid_agn.validation.simulator import (
    LineTruth,
    SpectrumTruth,
    simulate_arrays,
    sir_wavelength_grid,
)


def test_catalogue_hypotheses_cover_spe_and_phz():
    row = {
        "spe_gal_z": 1.31,
        "spe_gal_z_prob": 0.9,
        "spe_qso_z": 2.6,
        "phz_mode_1": 1.28,
        "phz_median": 1.4,
        "phz_mode_2": np.nan,
    }
    origins = {h.origin for h in from_catalogue(row)}
    assert origins == {"spe_galaxy", "spe_qso", "phz_mode_1", "phz_median"}


def test_catalogue_hypotheses_tolerate_a_bare_row():
    assert from_catalogue({}) == []
    assert from_catalogue({"spe_gal_z": -99.0}) == []


def test_probability_cut_is_applied_when_asked():
    row = {"spe_gal_z": 1.3, "spe_gal_z_prob": 0.1}
    assert from_catalogue(row) != []
    assert from_catalogue(row, min_probability=0.5) == []


def test_find_peaks_locates_injected_spikes():
    snr = np.zeros(200)
    snr[50] = 8.0
    snr[120] = 6.0
    snr[121] = 5.0
    peaks = find_peaks(np.arange(200.0), snr, threshold=4.0)
    assert 50 in peaks
    assert 120 in peaks
    assert 121 not in peaks  # suppressed by the separation rule


def test_find_peaks_ignores_noise_below_threshold():
    rng = np.random.default_rng(0)
    assert find_peaks(np.arange(500.0), rng.normal(size=500), threshold=4.0).size == 0


def test_peak_hypotheses_recover_the_true_redshift():
    z_true = 1.25
    truth = SpectrumTruth(
        z=z_true,
        continuum_flux=1e-17,
        noise_flux=2e-19,
        seed=1,
        lines=(
            LineTruth("Halpha", 3e-15, 200.0),
            LineTruth("NII6584", 1e-15, 200.0),
        ),
    )
    w, f, v, m, q = simulate_arrays(truth)
    hypotheses = from_peaks(w, f, v, threshold=5.0)
    separations = [
        C_KMS * abs(h.z - z_true) / (1 + z_true) for h in hypotheses if h.origin == "peaks"
    ]
    assert min(separations) < 500.0


def test_peak_hypotheses_require_a_second_line_by_default():
    w = sir_wavelength_grid()
    rng = np.random.default_rng(2)
    f = 1e-17 + rng.normal(0, 2e-19, w.size)
    f[300] += 5e-18  # one isolated spike
    v = np.full(w.size, (2e-19) ** 2)
    strict = from_peaks(w, f, v, threshold=4.0, require_second_line=True)
    loose = from_peaks(w, f, v, threshold=4.0, require_second_line=False)
    assert len(strict) < len(loose)


def test_blind_grid_is_uniform_in_velocity():
    grid = blind_grid(z_min=1.0, z_max=1.5, step_kms=1000.0)
    z = np.array([h.z for h in grid])
    steps = C_KMS * np.diff(z) / (1.0 + z[:-1])
    assert np.allclose(steps, 1000.0, rtol=0.02)
    assert all(h.origin == "blind" for h in grid)


def test_blind_grid_skips_the_one_untestable_redshift_interval():
    """Only one interval below z=5.6 has no permitted line in the red grism.

    Pa-delta leaves the red end at z = 0.840 and H-alpha enters at z = 0.904.
    Everywhere else some permitted line is available, so a BLR test is possible.
    """
    grid = blind_grid(z_min=0.0, z_max=5.7, step_kms=1000.0, require_broad=True)
    z = np.array([h.z for h in grid])
    assert not np.any((z > 0.85) & (z < 0.895)), sorted(z[(z > 0.85) & (z < 0.895)])
    for lo, hi in ((0.1, 0.4), (0.3, 0.7), (1.0, 1.7), (2.0, 2.7), (3.0, 3.4), (4.0, 5.4)):
        assert np.any((z > lo) & (z < hi)), f"no hypotheses in {lo}-{hi}"


def test_every_permitted_line_in_range_belongs_to_a_system():
    """A permitted line outside every system is a blind spot in the BLR branch."""
    from euclid_agn.models.line_catalog import LINES, SYSTEMS, redshift_range

    broad_capable = set()
    for system in SYSTEMS:
        broad_capable |= set(system.broad_members)
    for line in LINES:
        if not line.permitted:
            continue
        lo, hi = redshift_range(line)
        if hi < 0 or lo > 5.7:
            continue
        assert line.name in broad_capable, f"{line.name} is permitted and in range but unusable"


def test_blind_grid_rejects_a_bad_step():
    with pytest.raises(ValueError):
        blind_grid(step_kms=0.0)


def test_deduplicate_keeps_the_most_informative_origin():
    merged = deduplicate(
        [
            RedshiftHypothesis(1.2000, "blind"),
            RedshiftHypothesis(1.2001, "spe_galaxy"),
            RedshiftHypothesis(1.9, "blind"),
        ],
        tolerance_kms=200.0,
    )
    assert len(merged) == 2
    survivor = merged[0]
    assert survivor.origin == "spe_galaxy"
    assert "blind" in survivor.metadata["merged_origins"]


def test_deduplicate_leaves_well_separated_hypotheses_alone():
    merged = deduplicate(
        [RedshiftHypothesis(1.20, "blind"), RedshiftHypothesis(1.21, "blind")],
        tolerance_kms=200.0,
    )
    assert len(merged) == 2


def test_generate_combines_every_source_and_records_origin():
    z_true = 1.3
    truth = SpectrumTruth(
        z=z_true, continuum_flux=1e-17, noise_flux=2e-19, seed=3,
        lines=(LineTruth("Halpha", 3e-15, 200.0),),
    )
    w, f, v, m, q = simulate_arrays(truth)
    row = {"spe_gal_z": 1.2999, "phz_mode_1": 0.95}
    hypotheses = generate(w, f, v, row=row, blind_step_kms=2000.0)
    origins = {h.origin for h in hypotheses}
    assert "blind" in origins
    assert "spe_galaxy" in origins or "peaks" in origins
    assert all(0.0 <= h.z <= 5.7 for h in hypotheses)
    assert [h.z for h in hypotheses] == sorted(h.z for h in hypotheses)


def test_generate_works_without_any_catalogue_information():
    """A source the templates never classified must still be searchable."""
    truth = SpectrumTruth(z=1.4, continuum_flux=1e-17, noise_flux=2e-19, seed=4)
    w, f, v, m, q = simulate_arrays(truth)
    hypotheses = generate(w, f, v, row=None, blind_step_kms=3000.0)
    assert len(hypotheses) > 10
    assert {h.origin for h in hypotheses} == {"blind"}


def test_halpha_rest_wavelength_drives_the_peak_identification():
    w = sir_wavelength_grid()
    z = 1.2
    observed = BY_NAME["Halpha"].rest * (1 + z)
    assert w[0] < observed < w[-1]
