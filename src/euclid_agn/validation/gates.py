"""Trade-off of the Stage-0 signal gate: what it keeps and what it costs.

The gate has three knobs - the feature statistic threshold, the per-dither
excess required, and how many dithers must show it - and each is a choice
between purity of the identifications that pass and completeness of the
sample that passes.  Neither side of the trade-off is free to set by taste:
both are measured here against an external redshift, and the operating point
is recorded with the run.

The tables produced are the selection-function inputs for the gate itself:
``P(pass | data properties)`` is what the completeness of any later AGN
sample has to be multiplied by.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GateSettings:
    feature_min_dchi2: float = 25.0
    dither_sigma: float = 2.5
    min_dithers: int = 3

    def as_dict(self) -> dict[str, float]:
        return {"feature_min_dchi2": self.feature_min_dchi2, "dither_sigma": self.dither_sigma,
                "min_dithers": float(self.min_dithers)}


def n_detected(excess: Sequence[float], sigma: float) -> tuple[int, int]:
    """``(n_dithers_above_sigma, n_dithers_evaluable)`` from a list of excesses."""
    values = np.asarray([np.nan if e is None else e for e in excess], dtype=float)
    finite = values[np.isfinite(values)]
    return int(np.sum(finite > sigma)), int(finite.size)


def passes(row, settings: GateSettings) -> bool:
    """Does one object pass the gate?  ``min_dithers`` is capped at what is evaluable."""
    if float(row["feature_max_dchi2"]) < settings.feature_min_dchi2:
        return False
    detected, evaluable = n_detected(row["excess"], settings.dither_sigma)
    if evaluable == 0:
        return False
    return detected >= min(settings.min_dithers, evaluable)


def apply_gate(table: pd.DataFrame, settings: GateSettings) -> pd.Series:
    return table.apply(lambda r: passes(r, settings), axis=1)


def trade_off(
    table: pd.DataFrame,
    feature_thresholds: Sequence[float] = (15.0, 25.0, 50.0),
    sigmas: Sequence[float] = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0),
    min_dithers: Sequence[int] = (1, 2, 3),
    reference_detectable: str | None = "dchi2_Halpha",
    reference_min: float = 25.0,
) -> pd.DataFrame:
    """One row per gate setting with purity and completeness.

    ``purity``        agreement with the external redshift among passers;
    ``pass_fraction`` fraction of all objects passing;
    ``completeness``  fraction of objects whose reference line is detectable
                      (``reference_detectable`` > ``reference_min``) that pass -
                      the gate's cost on objects that genuinely have signal;
    ``n_correct_kept`` how many correct identifications survive.
    """
    rows = []
    has_signal = (
        table[reference_detectable] > reference_min
        if reference_detectable and reference_detectable in table
        else pd.Series(True, index=table.index)
    )
    for f in feature_thresholds:
        for s in sigmas:
            for k in min_dithers:
                settings = GateSettings(f, s, k)
                keep = apply_gate(table, settings)
                kept = table[keep]
                rows.append({
                    **settings.as_dict(),
                    "n_pass": int(keep.sum()),
                    "pass_fraction": float(keep.mean()),
                    "purity": float(kept["agrees"].mean()) if len(kept) else np.nan,
                    "completeness": float(keep[has_signal].mean()) if has_signal.any() else np.nan,
                    "n_correct_kept": int(kept["agrees"].sum()),
                    "correct_retained": float(kept["agrees"].sum() / table["agrees"].sum()) if table["agrees"].sum() else np.nan,
                })
    return pd.DataFrame(rows)


def pareto_front(table: pd.DataFrame, x: str = "completeness", y: str = "purity") -> pd.DataFrame:
    """Settings not dominated in both ``x`` and ``y``."""
    t = table.dropna(subset=[x, y]).sort_values(x, ascending=False)
    best_y = -np.inf; keep = []
    for index, row in t.iterrows():
        if row[y] > best_y:
            keep.append(index); best_y = row[y]
    return t.loc[keep].sort_values(x)
