"""Reference-data handling: sentinels, joins and velocity comparisons."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from euclid_agn.constants import C_KMS
from euclid_agn.validation.truth import (
    SPE_SENTINEL,
    agreement_summary,
    compare_redshifts,
    line_summary,
    spe_lines,
    spe_reference,
    velocity_difference,
)


class FakeTap:
    def __init__(self, frames):
        self.frames = list(frames)
        self.queries = []

    def query(self, adql):
        self.queries.append(adql)
        return self.frames.pop(0)


class FakeBackend:
    def __init__(self, frames):
        self.tap = FakeTap(frames)


def test_velocity_difference_is_symmetric_and_signed():
    assert velocity_difference(1.2, 1.2) == pytest.approx(0.0)
    forward = velocity_difference(1.201, 1.2)
    backward = velocity_difference(1.2, 1.201)
    assert forward == pytest.approx(-backward)
    assert forward == pytest.approx(C_KMS * 0.001 / 2.2005, rel=1e-6)


def test_sentinel_rows_are_discarded():
    """SPE writes -99 rather than NULL; it must never reach a mean."""
    frame = pd.DataFrame(
        {
            "object_id": [1, 2],
            "spe_rank": [0, 0],
            "spe_line_name": ["Halpha", "Halpha"],
            "spe_line_central_wl_gf": [14442.0, SPE_SENTINEL],
            "spe_line_flux_gf": [1e-16, SPE_SENTINEL],
            "spe_line_snr_gf": [12.0, SPE_SENTINEL],
            "spe_line_fwhm_gf": [30.0, SPE_SENTINEL],
        }
    )
    out = spe_lines(FakeBackend([frame]), [1, 2])
    assert list(out["object_id"]) == [1]


def test_line_summary_picks_the_strongest_line_per_object():
    lines = pd.DataFrame(
        {
            "object_id": [1, 1, 2],
            "spe_line_name": ["Halpha", "NII6584", "Hbeta"],
            "spe_line_snr_gf": [12.0, 4.0, 7.0],
            "spe_line_central_wl_gf": [14442.0, 14490.0, 15000.0],
        }
    )
    summary = line_summary(lines).set_index("object_id")
    assert summary.loc[1, "spe_best_line"] == "Halpha"
    assert summary.loc[1, "n_spe_lines"] == 2
    assert summary.loc[2, "n_spe_lines"] == 1


def test_line_summary_handles_no_detections():
    assert line_summary(pd.DataFrame()).empty


def test_spe_reference_marks_objects_without_lines():
    redshifts = pd.DataFrame(
        {"object_id": [1], "spe_z": [1.2], "spe_z_err": [0.001], "spe_z_prob": [0.9],
         "spe_cont_snr": [5.0]}
    ).rename(columns={"spe_z": "spe_gal_z", "spe_z_err": "spe_gal_z_err",
                      "spe_z_prob": "spe_gal_z_prob"})
    classification = pd.DataFrame(
        {"object_id": [1], "spe_class": ["galaxy"], "spe_gal_prob": [0.9], "spe_qso_prob": [0.1]}
    )
    empty_lines = pd.DataFrame(
        columns=["object_id", "spe_rank", "spe_line_name", "spe_line_central_wl_gf",
                 "spe_line_flux_gf", "spe_line_snr_gf", "spe_line_fwhm_gf"]
    )
    backend = FakeBackend([redshifts, classification, empty_lines])
    out = spe_reference(backend, [1])
    assert out["n_spe_lines"].iloc[0] == 0


def test_compare_redshifts_flags_agreement_within_tolerance():
    results = pd.DataFrame({"object_id": [1, 2, 3], "z": [1.2000, 1.2100, 0.5]})
    reference = pd.DataFrame({"object_id": [1, 2, 3], "spe_gal_z": [1.2001, 1.2000, np.nan]})
    compared = compare_redshifts(results, reference, tolerance_kms=1000.0)
    assert len(compared) == 2  # the NaN reference is dropped
    assert bool(compared.set_index("object_id").loc[1, "agrees"])
    assert not bool(compared.set_index("object_id").loc[2, "agrees"])


def test_agreement_summary_can_split_by_a_column():
    compared = pd.DataFrame(
        {
            "object_id": [1, 2, 3, 4],
            "delta_v_kms": [10.0, -20.0, 5000.0, 8000.0],
            "agrees": [True, True, False, False],
            "bin": ["strong", "strong", "weak", "weak"],
        }
    )
    overall = agreement_summary(compared)
    assert overall["fraction_agreeing"].iloc[0] == 0.5
    split = agreement_summary(compared, by="bin").set_index("bin")
    assert split.loc["strong", "fraction_agreeing"] == 1.0
    assert split.loc["weak", "fraction_agreeing"] == 0.0


def test_agreement_summary_of_nothing():
    assert agreement_summary(pd.DataFrame()).empty
