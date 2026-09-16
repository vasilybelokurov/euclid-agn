"""The blind-redshift experiment must recover a known redshift."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from euclid_agn.constants import C_KMS
from euclid_agn.fit.screen import ScreenSettings
from euclid_agn.validation.metrics import (
    blind_best_redshift,
    blind_redshift_experiment,
    catastrophic_fraction,
)
from euclid_agn.validation.simulator import (
    LineTruth,
    SpectrumTruth,
    simulate_observation,
    write_sir_file,
)

SETTINGS = ScreenSettings(broad_sigma_kms=(600.0, 1500.0), n_refine=0)
NARROW = (
    LineTruth("Halpha", 6e-16, 150.0),
    LineTruth("NII6584", 2e-16, 150.0),
    LineTruth("SII6716", 1e-16, 150.0),
    LineTruth("SII6731", 1e-16, 150.0),
)


@pytest.fixture(scope="module")
def archive(tmp_path_factory):
    directory = tmp_path_factory.mktemp("metrics")
    observations = [
        simulate_observation(
            SpectrumTruth(z=1.20, lines=NARROW, object_id=301, seed=1, noise_flux=1e-19)
        ),
        simulate_observation(
            SpectrumTruth(z=1.45, lines=NARROW, object_id=302, seed=2, noise_flux=1e-19)
        ),
        # A featureless spectrum: the blind scan has nothing to lock onto.
        simulate_observation(
            SpectrumTruth(z=1.20, lines=(), object_id=303, seed=3, noise_flux=1e-19)
        ),
    ]
    path = write_sir_file(directory / "EUC_SIR_W-COMBSPEC_3_sim.fits", observations)
    return [str(path)]


@pytest.fixture(scope="module")
def reference():
    return pd.DataFrame(
        {
            "object_id": [301, 302, 303],
            "spe_gal_z": [1.20, 1.45, 1.20],
            "spe_best_snr": [30.0, 12.0, 1.0],
        }
    )


def test_blind_scan_recovers_a_strong_line_redshift(archive):
    from euclid_agn.io.sir import open_sir_file

    with open_sir_file(archive[0]) as sir:
        spectrum = sir.read_combined(sir.group_for_object(301))
    best = blind_best_redshift(spectrum, SETTINGS, step_kms=400.0)
    assert best is not None
    assert abs(C_KMS * (best["z"] - 1.20) / 2.20) < 1000.0
    assert best["system"] == "halpha_complex"
    assert best["n_narrow_lines"] >= 4


def test_blind_scan_uses_no_catalogue_information(archive):
    """Two objects at different redshifts must give different answers."""
    from euclid_agn.io.sir import open_sir_file

    answers = {}
    with open_sir_file(archive[0]) as sir:
        for object_id in (301, 302):
            spectrum = sir.read_combined(sir.group_for_object(object_id))
            answers[object_id] = blind_best_redshift(spectrum, SETTINGS, step_kms=400.0)["z"]
    assert abs(answers[301] - 1.20) < 0.01
    assert abs(answers[302] - 1.45) < 0.01


def test_experiment_reports_agreement_and_stratifies_by_line_strength(archive, reference):
    result = blind_redshift_experiment(
        archive, reference, settings=SETTINGS, step_kms=400.0, tolerance_kms=1000.0
    )
    assert len(result.compared) == 3
    agreeing = result.compared.set_index("object_id")["agrees"]
    assert bool(agreeing.loc[301])
    assert bool(agreeing.loc[302])
    summary = result.summary()
    assert 0.0 <= summary["fraction_agreeing"].iloc[0] <= 1.0
    assert "snr_bin" in result.compared.columns


def test_catastrophic_fraction_matches_the_agreement_flag(archive, reference):
    result = blind_redshift_experiment(
        archive, reference, settings=SETTINGS, step_kms=400.0, tolerance_kms=1000.0
    )
    assert catastrophic_fraction(result.compared, 1000.0) == pytest.approx(
        1.0 - result.compared["agrees"].mean()
    )


def test_experiment_skips_objects_with_no_reference(archive):
    empty = pd.DataFrame({"object_id": [999], "spe_gal_z": [1.0]})
    result = blind_redshift_experiment(archive, empty, settings=SETTINGS, step_kms=1000.0)
    assert result.table.empty
    assert result.compared.empty


def test_catastrophic_fraction_of_nothing_is_nan():
    assert np.isnan(catastrophic_fraction(pd.DataFrame()))


def test_nineteen_digit_object_ids_survive_the_reference_lookup(archive):
    """float64 cannot hold a MER object id; the lookup must key on int64."""
    big = 2708573889636910920
    assert int(float(big)) != big  # the trap
    from euclid_agn.io.sir import open_sir_file
    from euclid_agn.validation.simulator import SpectrumTruth, simulate_observation, write_sir_file
    import tempfile, pathlib

    with tempfile.TemporaryDirectory() as d:
        path = write_sir_file(
            pathlib.Path(d) / "big.fits",
            [simulate_observation(SpectrumTruth(z=1.2, lines=NARROW, object_id=big, seed=9, noise_flux=1e-19))],
        )
        reference = pd.DataFrame({"object_id": [big], "spe_gal_z": [1.2], "phz_mode_1": [1.22], "spe_best_snr": [10.0]})
        result = blind_redshift_experiment(
            [str(path)], reference, settings=SETTINGS, step_kms=600.0, prior_column="phz_mode_1"
        )
        assert len(result.compared) == 1
        assert int(result.compared["object_id"].iloc[0]) == big
