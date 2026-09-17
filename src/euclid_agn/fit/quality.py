"""Redshift-quality warning bits, in the manner of Redrock's ``ZWARN``.

DESI's practice is the right one: a redshift comes with a bitmask saying what
might be wrong with it, and a clean sample is ``zwarn == 0`` plus a threshold
on the best-minus-runner-up chi-squared.  The bits below are the conditions
this pipeline has actually seen produce wrong redshifts on real Q1 spectra,
each verified against DESI:

======================  ==========================================================
bit                     condition
======================  ==========================================================
SMALL_DELTA_CHI2        best-minus-best-other-system statistic below threshold
NO_LINE                 identification statistic too small: no line detected
BAD_CONTINUUM           reduced chi-squared of M0 far from its expectation
OUTLIER_PIXELS          many spike pixels rejected: the spectrum is damaged
LOW_USABLE_FRACTION     too few usable pixels in the science window
NO_PRIOR                no photometric redshift was available to break degeneracy
PRIOR_DECIDED           the prior, not the data, separated the winner from the
                        runner-up
CONTAMINATED            many contaminating sources listed in the dithers
EDGE_LINE               the winning line sits near the edge of the covered range
======================  ==========================================================

Bit values are stable: never renumber, only append.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntFlag

import numpy as np
import pandas as pd


class ZWarn(IntFlag):
    NONE = 0
    SMALL_DELTA_CHI2 = 1
    NO_LINE = 2
    BAD_CONTINUUM = 4
    OUTLIER_PIXELS = 8
    LOW_USABLE_FRACTION = 16
    NO_PRIOR = 32
    PRIOR_DECIDED = 64
    CONTAMINATED = 128
    EDGE_LINE = 256
    INCOHERENT_DITHERS = 512  # the strongest feature is not present in >= 2 dithers
    NO_FEATURE = 1024  # no line-shaped feature anywhere, redshift-agnostic
    AMBIGUOUS_CLASS = 2048  # best class beats the runner-up class by less than the threshold
    MODELS_DISAGREE = 4096  # lines+spline and continuum+lines models give different redshifts


@dataclass(frozen=True)
class QualityThresholds:
    """Every number a quality bit depends on, so it lands in the run manifest.

    ``min_margin`` is the Redrock-style best-minus-runner-up threshold.  It is
    *calibrated*, not chosen: read from purity against DESI on the holdout.
    """

    min_margin: float = 10.0
    min_identification: float = 25.0
    max_chi2_reduced_m0: float = 4.0
    max_outlier_pixels: int = 8
    min_usable_fraction: float = 0.5
    max_contaminants_per_dither: float = 12.0
    edge_margin_pixels: float = 6.0
    #: Stage-0 feature gate: strongest single-line statistic anywhere.  The
    #: null maximum over ~450 positions for one component is ~7 + 6.6 (see
    #: JOURNAL); 25 is well clear of it.
    min_feature_dchi2: float = 25.0


def zwarn_for_row(row, thresholds: QualityThresholds = QualityThresholds()) -> int:
    """Compute the warning bitmask for one screening row.

    Missing columns never raise: a bit is set only when its condition can be
    evaluated and is met, except ``NO_PRIOR``, which is set when the prior
    column is absent or NaN.
    """
    t = thresholds
    warn = ZWarn.NONE

    def get(name, default=np.nan):
        try:
            value = row[name]
        except (KeyError, IndexError, TypeError):
            return default
        return default if value is None else value

    margin = float(get("delta_chi2_over_other_system"))
    if np.isfinite(margin) and margin < t.min_margin:
        warn |= ZWarn.SMALL_DELTA_CHI2

    ident = float(get("quick_delta_chi2_identification", get("delta_chi2_identification")))
    if np.isfinite(ident) and ident < t.min_identification:
        warn |= ZWarn.NO_LINE

    chi2 = float(get("chi2_reduced_m0"))
    if np.isfinite(chi2) and chi2 > t.max_chi2_reduced_m0:
        warn |= ZWarn.BAD_CONTINUUM

    outliers = float(get("n_outlier_pixels"))
    if np.isfinite(outliers) and outliers > t.max_outlier_pixels:
        warn |= ZWarn.OUTLIER_PIXELS

    usable = float(get("quality_usable_pixel_fraction", get("usable_pixel_fraction")))
    if np.isfinite(usable) and usable < t.min_usable_fraction:
        warn |= ZWarn.LOW_USABLE_FRACTION

    prior = float(get("z_prior"))
    if not np.isfinite(prior):
        warn |= ZWarn.NO_PRIOR
    penalty_gap = float(get("prior_penalty"))
    # If removing the prior penalty from the winner would hand the win to the
    # runner-up, the prior decided.  Only computable when both are recorded.
    data_margin = float(get("data_margin"))
    if np.isfinite(data_margin) and data_margin <= 0.0 and np.isfinite(penalty_gap):
        warn |= ZWarn.PRIOR_DECIDED

    contaminants = float(get("contaminants_max_per_dither"))
    if np.isfinite(contaminants) and contaminants > t.max_contaminants_per_dither:
        warn |= ZWarn.CONTAMINATED

    edge = float(get("winning_line_edge_pixels"))
    if np.isfinite(edge) and edge < t.edge_margin_pixels:
        warn |= ZWarn.EDGE_LINE

    feature = float(get("feature_max_dchi2"))
    if np.isfinite(feature) and feature < t.min_feature_dchi2:
        warn |= ZWarn.NO_FEATURE
    coherent = get("feature_coherent", None)
    evaluable = float(get("feature_n_dithers_evaluable"))
    if coherent is not None and np.isfinite(evaluable) and evaluable >= 1 and not bool(float(coherent)):
        warn |= ZWarn.INCOHERENT_DITHERS

    return int(warn)


def add_zwarn(table: pd.DataFrame, thresholds: QualityThresholds = QualityThresholds()) -> pd.DataFrame:
    """Append ``zwarn`` and ``zwarn_names`` columns to a screening table."""
    if table.empty:
        return table
    out = table.copy()
    out["zwarn"] = [zwarn_for_row(row, thresholds) for _, row in out.iterrows()]
    out["zwarn_names"] = [describe(int(v)) for v in out["zwarn"]]
    return out


def describe(value: int) -> str:
    names = [flag.name for flag in ZWarn if flag and value & flag]
    return "|".join(names) if names else "OK"


def purity_table(compared: pd.DataFrame, margins=(0, 5, 10, 15, 20, 30), require_clean: bool = True) -> pd.DataFrame:
    """Purity and retained fraction against the margin threshold.

    ``compared`` needs ``agrees`` and ``delta_chi2_over_other_system``; with
    ``require_clean`` it also needs ``zwarn`` and considers only rows whose
    other bits (everything except SMALL_DELTA_CHI2) are clear.  This is the
    table the ``min_margin`` threshold is read from.
    """
    rows = []
    total_correct = int(compared["agrees"].sum())
    base = compared
    if require_clean and "zwarn" in compared:
        other = compared["zwarn"] & ~int(ZWarn.SMALL_DELTA_CHI2)
        base = compared[other == 0]
    for margin in margins:
        kept = base[base["delta_chi2_over_other_system"] > margin]
        rows.append({"margin": margin, "n": len(kept),
                     "purity": float(kept["agrees"].mean()) if len(kept) else float("nan"),
                     "retained_of_correct": kept["agrees"].sum() / total_correct if total_correct else float("nan")})
    return pd.DataFrame(rows)
