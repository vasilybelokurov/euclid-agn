import numpy as np
import pandas as pd
import pytest

from euclid_agn.validation.gates import GateSettings, apply_gate, n_detected, pareto_front, passes, trade_off


def test_n_detected_ignores_unevaluable_dithers():
    assert n_detected([5.0, None, 1.0, np.nan], 3.0) == (1, 2)
    assert n_detected([], 3.0) == (0, 0)


def test_passes_requires_feature_and_coherence_capped_by_evaluable():
    row = {"feature_max_dchi2": 60.0, "excess": [5.0, 4.0, 0.5, None]}
    assert passes(row, GateSettings(25.0, 3.0, 2))
    assert not passes(row, GateSettings(25.0, 3.0, 3))
    assert not passes({"feature_max_dchi2": 10.0, "excess": [5.0, 4.0]}, GateSettings(25.0, 3.0, 2))
    # a single evaluable dither: min_dithers is capped at 1
    assert passes({"feature_max_dchi2": 60.0, "excess": [5.0, None, None]}, GateSettings(25.0, 3.0, 2))
    assert not passes({"feature_max_dchi2": 60.0, "excess": [None, None]}, GateSettings(25.0, 3.0, 1))


def sample():
    return pd.DataFrame({
        "feature_max_dchi2": [80.0, 80.0, 80.0, 10.0, 80.0],
        "excess": [[6.0, 5.0, 4.0], [9.0, 0.5, 0.2], [4.0, 3.5, None], [1.0, 1.0, 1.0], [2.5, 2.2, 2.4]],
        "agrees": [True, False, True, False, True],
        "dchi2_Halpha": [100.0, 5.0, 60.0, 3.0, 40.0],
    })


def test_trade_off_moves_the_right_way():
    t = trade_off(sample(), feature_thresholds=(25.0,), sigmas=(2.0, 3.0), min_dithers=(1, 2))
    strict = t[(t.dither_sigma == 3.0) & (t.min_dithers == 2)].iloc[0]
    loose = t[(t.dither_sigma == 2.0) & (t.min_dithers == 1)].iloc[0]
    assert strict["purity"] >= loose["purity"]
    assert strict["completeness"] <= loose["completeness"]
    # the single-dither contaminant (row 1) is removed by the coherence requirement
    assert strict["n_pass"] == 2 and strict["purity"] == 1.0


def test_pareto_front_keeps_undominated_settings():
    # (0.6, 0.6) is dominated by (0.7, 0.7); everything else is on the front
    t = pd.DataFrame({"completeness": [0.9, 0.7, 0.5, 0.8, 0.6], "purity": [0.5, 0.7, 0.9, 0.6, 0.6]})
    front = pareto_front(t)
    assert set(front.index) == {0, 1, 2, 3}


def test_apply_gate_returns_boolean_series():
    keep = apply_gate(sample(), GateSettings())
    assert keep.dtype == bool and keep.sum() == 2  # rows 0 and 2 (row 2 has two evaluable dithers, both above)
