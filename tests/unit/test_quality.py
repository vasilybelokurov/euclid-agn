import numpy as np
import pandas as pd
import pytest

from euclid_agn.fit.quality import QualityThresholds, ZWarn, add_zwarn, describe, purity_table, zwarn_for_row


def clean_row(**overrides):
    row = {"delta_chi2_over_other_system": 40.0, "quick_delta_chi2_identification": 200.0, "chi2_reduced_m0": 2.0,
           "n_outlier_pixels": 1, "quality_usable_pixel_fraction": 0.95, "z_prior": 1.2, "prior_penalty": 0.1,
           "data_margin": 35.0, "contaminants_max_per_dither": 4, "winning_line_edge_pixels": 100.0}
    row.update(overrides)
    return pd.Series(row)


def test_clean_row_has_no_warnings():
    assert zwarn_for_row(clean_row()) == 0
    assert describe(0) == "OK"


@pytest.mark.parametrize("field,value,bit", [
    ("delta_chi2_over_other_system", 3.0, ZWarn.SMALL_DELTA_CHI2),
    ("quick_delta_chi2_identification", 5.0, ZWarn.NO_LINE),
    ("chi2_reduced_m0", 9.0, ZWarn.BAD_CONTINUUM),
    ("n_outlier_pixels", 20, ZWarn.OUTLIER_PIXELS),
    ("quality_usable_pixel_fraction", 0.3, ZWarn.LOW_USABLE_FRACTION),
    ("z_prior", np.nan, ZWarn.NO_PRIOR),
    ("data_margin", -2.0, ZWarn.PRIOR_DECIDED),
    ("contaminants_max_per_dither", 20, ZWarn.CONTAMINATED),
    ("winning_line_edge_pixels", 2.0, ZWarn.EDGE_LINE),
])
def test_each_condition_sets_exactly_its_bit(field, value, bit):
    warn = zwarn_for_row(clean_row(**{field: value}))
    assert warn == int(bit)
    assert describe(warn) == bit.name


def test_bits_combine_and_missing_columns_do_not_raise():
    warn = zwarn_for_row(pd.Series({"delta_chi2_over_other_system": 1.0, "chi2_reduced_m0": 10.0}))
    assert warn & ZWarn.SMALL_DELTA_CHI2 and warn & ZWarn.BAD_CONTINUUM and warn & ZWarn.NO_PRIOR
    assert "SMALL_DELTA_CHI2" in describe(warn) and "BAD_CONTINUUM" in describe(warn)


def test_bit_values_are_stable():
    assert int(ZWarn.SMALL_DELTA_CHI2) == 1 and int(ZWarn.NO_LINE) == 2 and int(ZWarn.EDGE_LINE) == 256


def test_thresholds_are_respected():
    strict = QualityThresholds(min_margin=50.0)
    assert zwarn_for_row(clean_row(), strict) & ZWarn.SMALL_DELTA_CHI2


def test_add_zwarn_and_purity_table():
    table = pd.DataFrame({
        "agrees": [True, True, False, False, True],
        "delta_chi2_over_other_system": [40.0, 12.0, 40.0, 2.0, 8.0],
        "quick_delta_chi2_identification": [200.0, 200.0, 200.0, 200.0, 200.0],
        "chi2_reduced_m0": [2.0, 2.0, 9.0, 2.0, 2.0],
        "z_prior": [1.0] * 5,
    })
    out = add_zwarn(table)
    assert out.loc[2, "zwarn"] & ZWarn.BAD_CONTINUUM
    clean = purity_table(out, margins=(0, 10), require_clean=True)
    # the BAD_CONTINUUM wrong answer is excluded before the margin cut
    assert clean.loc[clean.margin == 0, "purity"].iloc[0] == pytest.approx(3 / 4)
    assert clean.loc[clean.margin == 10, "purity"].iloc[0] == pytest.approx(1.0)
    assert clean.loc[clean.margin == 10, "retained_of_correct"].iloc[0] == pytest.approx(2 / 3)
    assert add_zwarn(pd.DataFrame()).empty
