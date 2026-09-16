"""External reference data for validating the pipeline.

Two kinds of external information, used for *evaluation only* and never as an
input filter:

``spe``   Euclid's own SPE redshifts and line measurements.  Not truth - SPE is
          a template fit with its own failure modes - but an independent
          measurement of the same photons, which makes it the right first check
          that the machinery recovers real lines.
``desi``  DESI spectroscopy, the external truth for the EDF-N validation set.

The rule from the project brief holds: these labels are held out, partitioned
before any threshold is chosen, and never used to select candidates.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd

from euclid_agn.archive import schema
from euclid_agn.archive.irsa import ID_CHUNK, IrsaQ1Backend
from euclid_agn.archive.tap import in_list_clause, quote_columns
from euclid_agn.constants import C_KMS

log = logging.getLogger(__name__)

#: Line-feature columns worth carrying.
LINE_COLUMNS: tuple[str, ...] = (
    "object_id",
    "spe_rank",
    "spe_line_name",
    "spe_line_central_wl_gf",
    "spe_line_flux_gf",
    "spe_line_snr_gf",
    "spe_line_fwhm_gf",
)

#: SPE marks unmeasured quantities with this sentinel rather than NULL.
SPE_SENTINEL = -99.0

#: Widest Gaussian FWHM a *narrow* line can plausibly have in the red grism.
#: The instrumental FWHM is ~32 A for a compact source and ~60 A for the most
#: extended ones.  VERIFIED on cached Q1 spectra: two SPE "detections" at
#: S/N 13.8 and 43.5 with FWHM 92 A and 143 A sit on pixels that are flat or
#: pure noise.  SPE line S/N is therefore not usable as truth without this
#: width check.
MAX_PLAUSIBLE_LINE_FWHM_ANGSTROM = 80.0


def _chunks(values: Sequence[int], size: int = ID_CHUNK) -> list[list[int]]:
    values = list(values)
    return [values[i : i + size] for i in range(0, len(values), size)]


def spe_redshifts(backend: IrsaQ1Backend, object_ids: Sequence[int]) -> pd.DataFrame:
    """Rank-0 SPE galaxy and QSO redshifts with their reliability columns."""
    frames = []
    for chunk in _chunks(object_ids):
        where = [in_list_clause("object_id", chunk), "spe_rank = 0"]
        galaxy = backend.tap.query(
            "SELECT "
            + quote_columns(["object_id", "spe_z", "spe_z_err", "spe_z_prob", "spe_cont_snr"])
            + f" FROM {schema.SPE_GALAXY_CANDIDATES} WHERE " + " AND ".join(where)
        ).rename(
            columns={
                "spe_z": "spe_gal_z",
                "spe_z_err": "spe_gal_z_err",
                "spe_z_prob": "spe_gal_z_prob",
            }
        )
        classification = backend.tap.query(
            "SELECT "
            + quote_columns(["object_id", "spe_class", "spe_gal_prob", "spe_qso_prob"])
            + f" FROM {schema.SPE_CLASSIFICATION} WHERE "
            + in_list_clause("object_id", chunk)
        )
        frames.append(galaxy.merge(classification, on="object_id", how="outer"))
    if not frames:
        return pd.DataFrame(columns=["object_id"])
    return pd.concat(frames, ignore_index=True)


def spe_lines(
    backend: IrsaQ1Backend, object_ids: Sequence[int], min_snr: float = 3.0
) -> pd.DataFrame:
    """Rank-0 SPE line detections above a signal-to-noise threshold.

    SPE writes ``-99`` rather than NULL for quantities it did not measure, so
    the sentinel is filtered here rather than being allowed to propagate into
    a mean somewhere downstream.
    """
    frames = []
    for chunk in _chunks(object_ids):
        frames.append(
            backend.tap.query(
                "SELECT "
                + quote_columns(list(LINE_COLUMNS))
                + f" FROM {schema.SPE_LINE_FEATURES} WHERE "
                + in_list_clause("object_id", chunk)
                + f" AND spe_rank = 0 AND spe_line_snr_gf > {min_snr}"
            )
        )
    if not frames:
        return pd.DataFrame(columns=list(LINE_COLUMNS))
    table = pd.concat(frames, ignore_index=True)
    for column in ("spe_line_central_wl_gf", "spe_line_flux_gf", "spe_line_snr_gf"):
        table = table[table[column] > SPE_SENTINEL + 1.0]
    table = table.reset_index(drop=True)
    fwhm = table["spe_line_fwhm_gf"]
    table["plausible_width"] = (fwhm > 0.0) & (fwhm <= MAX_PLAUSIBLE_LINE_FWHM_ANGSTROM)
    return table


def line_summary(lines: pd.DataFrame) -> pd.DataFrame:
    """One row per object: how many lines SPE found and how strong the best is."""
    if lines.empty:
        return pd.DataFrame(columns=["object_id", "n_spe_lines", "spe_best_line", "spe_best_snr"])
    best = lines.loc[lines.groupby("object_id")["spe_line_snr_gf"].idxmax()]
    counts = lines.groupby("object_id").size().rename("n_spe_lines")
    return (
        best[["object_id", "spe_line_name", "spe_line_snr_gf", "spe_line_central_wl_gf"]]
        .rename(
            columns={
                "spe_line_name": "spe_best_line",
                "spe_line_snr_gf": "spe_best_snr",
                "spe_line_central_wl_gf": "spe_best_wavelength",
            }
        )
        .merge(counts, on="object_id")
        .reset_index(drop=True)
    )


def spe_reference(
    backend: IrsaQ1Backend,
    object_ids: Sequence[int],
    min_snr: float = 3.0,
    require_plausible_width: bool = True,
) -> pd.DataFrame:
    """Redshifts, classification and line summary for a list of objects.

    With ``require_plausible_width`` (the default) SPE lines wider than
    :data:`MAX_PLAUSIBLE_LINE_FWHM_ANGSTROM` are dropped before the summary, so
    ``spe_best_snr`` refers to a line that can actually be a line.
    """
    redshifts = spe_redshifts(backend, object_ids)
    lines = spe_lines(backend, object_ids, min_snr=min_snr)
    if require_plausible_width and not lines.empty:
        lines = lines[lines["plausible_width"]]
    summary = line_summary(lines)
    if summary.empty:
        redshifts["n_spe_lines"] = 0
        return redshifts
    return redshifts.merge(summary, on="object_id", how="left").fillna({"n_spe_lines": 0})


def velocity_difference(z_a, z_b) -> np.ndarray:
    """Velocity separation between two redshifts, km/s, in the mean frame."""
    z_a = np.asarray(z_a, dtype=float)
    z_b = np.asarray(z_b, dtype=float)
    return C_KMS * (z_a - z_b) / (1.0 + 0.5 * (z_a + z_b))


def compare_redshifts(
    results: pd.DataFrame,
    reference: pd.DataFrame,
    reference_column: str = "spe_gal_z",
    tolerance_kms: float = 1000.0,
) -> pd.DataFrame:
    """Join pipeline redshifts to a reference and measure the disagreement.

    ``agrees`` is a *comparison*, not a verdict: where the pipeline and SPE
    disagree, either can be at fault, and the identification degeneracies in
    this grism make disagreement expected for weak-line objects.
    """
    columns = [c for c in reference.columns if c == "object_id" or c not in results.columns]
    if reference_column not in columns:
        columns.append(reference_column)
    merged = results.merge(
        reference[columns].dropna(subset=[reference_column]), on="object_id", how="inner"
    )
    merged["delta_v_kms"] = velocity_difference(merged["z"], merged[reference_column])
    merged["agrees"] = np.abs(merged["delta_v_kms"]) < tolerance_kms
    return merged


def agreement_summary(compared: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    """Fraction agreeing, optionally split by a column such as line strength."""
    if compared.empty:
        return pd.DataFrame()
    if by is None:
        return pd.DataFrame(
            [
                {
                    "n": len(compared),
                    "fraction_agreeing": float(compared["agrees"].mean()),
                    "median_abs_delta_v_kms": float(np.median(np.abs(compared["delta_v_kms"]))),
                }
            ]
        )
    grouped = compared.groupby(by)
    return pd.DataFrame(
        {
            "n": grouped.size(),
            "fraction_agreeing": grouped["agrees"].mean(),
            "median_abs_delta_v_kms": grouped["delta_v_kms"].apply(
                lambda s: float(np.median(np.abs(s)))
            ),
        }
    ).reset_index()
