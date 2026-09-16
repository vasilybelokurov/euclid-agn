import numpy as np
import pytest

from euclid_agn.spectra.masking import (
    MASK_BITS,
    MaskDefinition,
    bit_is_set,
    mask_bit_fractions,
    usable_pixels,
)


def test_bit_values_match_the_release():
    assert MASK_BITS["NOT_USE"] == 1
    assert MASK_BITS["LOW_SNR"] == 2
    assert MASK_BITS["ABS_FLUX"] == 64


def test_real_mask_values_decompose_as_bitfields():
    # Values actually seen in tile 102160339: 1, 64, 65, 66, 67.
    mask = np.array([0, 1, 64, 65, 66, 67])
    assert bit_is_set(mask, 1).tolist() == [False, True, False, True, False, True]
    assert bit_is_set(mask, 2).tolist() == [False, False, False, False, True, True]
    assert bit_is_set(mask, 64).tolist() == [False, False, True, True, True, True]


def test_usable_rejects_not_use_only_by_default():
    mask = np.array([0, 1, 2, 64, 65])
    assert usable_pixels(mask).tolist() == [True, False, True, True, False]


def test_usable_can_reject_more_bits():
    mask = np.array([0, 1, 2, 64])
    got = usable_pixels(mask, reject=("NOT_USE", "LOW_SNR"))
    assert got.tolist() == [True, False, False, True]


def test_mask_definition_from_header_prefers_file_values():
    header = {"HIERARCH MSK_FLAG_NOT_USE": 1, "MSK_FLAG_ODD": 128}
    header = {"MSK_FLAG_NOT_USE": 1, "MSK_FLAG_ODD": 128}
    definition = MaskDefinition.from_header(header)
    assert definition.value("ODD") == 128
    with pytest.raises(KeyError):
        definition.value("LOW_SNR")


def test_mask_fractions_sum_sensibly():
    mask = np.array([0, 0, 1, 3])
    fractions = mask_bit_fractions(mask)
    assert fractions["NOT_USE"] == 0.5
    assert fractions["LOW_SNR"] == 0.25
    assert fractions["GOOD"] == 0.5
