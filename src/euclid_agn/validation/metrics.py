"""Validation experiments and their metrics.

The first experiment that can be run without any injection machinery is
**blind redshift recovery**: give the pipeline no catalogue information at all,
let it scan redshift space, and ask whether it lands where Euclid's own SPE
template fit landed.

It is a genuine test of the whole chain - IO, masking, continuum, line
catalogue, LSF, matched filter, ranking - because agreement requires every part
to be right at once, and it uses an independent measurement rather than a
simulation.  It is *not* a test of AGN detection: SPE and this pipeline see the
same photons, so agreement says the lines are real and the fit is useful, not
that the object is an AGN.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from euclid_agn.fit.hypotheses import blind_grid, deduplicate, from_catalogue
from euclid_agn.fit.screen import ScreenSettings, quick_scan, refine_redshifts_locally
from euclid_agn.io.sir import open_sir_file
from euclid_agn.validation.truth import agreement_summary, compare_redshifts

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class BlindRedshiftResult:
    """Per-object outcome of the blind scan."""

    table: pd.DataFrame
    compared: pd.DataFrame

    def summary(self, by: str | None = None) -> pd.DataFrame:
        return agreement_summary(self.compared, by=by)


def blind_best_redshift(
    spectrum,
    settings: ScreenSettings,
    hypotheses=None,
    z_min: float = 0.0,
    z_max: float = 5.7,
    step_kms: float = 400.0,
) -> dict | None:
    """Best redshift from a blind scan, using all the line evidence.

    No catalogue redshift enters this function: that is the point.
    """
    hypotheses = hypotheses or blind_grid(z_min=z_min, z_max=z_max, step_kms=step_kms)
    scan = quick_scan(spectrum, hypotheses, settings)
    if scan.empty:
        return None
    scan = refine_redshifts_locally(spectrum, scan, settings)
    best = scan.loc[scan["delta_chi2_penalised"].idxmax()]
    runner_up = scan[scan["system"] != best["system"]]
    margin = (
        float(best["delta_chi2_penalised"] - runner_up["delta_chi2_penalised"].max())
        if not runner_up.empty
        else float("inf")
    )
    return {
        "z": float(best["z"]),
        "origin": str(best["origin"]),
        "system": str(best["system"]),
        "n_narrow_lines": int(best["n_narrow_lines"]),
        "delta_chi2_narrow": float(best["delta_chi2_narrow"]),
        "delta_chi2_broad": float(best["delta_chi2_broad"]),
        "delta_chi2_total": float(best["delta_chi2_total"]),
        "delta_chi2_penalised": float(best["delta_chi2_penalised"]),
        "n_components": int(best["n_components"]),
        "delta_chi2_over_other_system": margin,
    }


def blind_redshift_experiment(
    files: Sequence[str],
    reference: pd.DataFrame,
    settings: ScreenSettings | None = None,
    step_kms: float = 400.0,
    tolerance_kms: float = 1000.0,
    min_usable_fraction: float = 0.5,
    max_objects: int | None = None,
    reference_column: str = "spe_gal_z",
    include_catalogue_hypotheses: bool = False,
) -> BlindRedshiftResult:
    """Run the blind scan over cached files and compare with a reference.

    Only data availability restricts the sample: a minimum usable-pixel
    fraction, and the existence of a reference redshift to compare against.
    No host property and no line-strength criterion enters the selection.

    With ``include_catalogue_hypotheses`` the SPE and PHZ redshifts of each
    object are added to the blind grid, as the production pipeline does.  That
    is no longer a blind test of the machinery - agreement with SPE is then
    partly by construction - but it measures what production will do, and the
    origin of the winning hypothesis is recorded so the two can be separated.
    """
    settings = settings or ScreenSettings(n_refine=0)
    hypotheses = blind_grid(step_kms=step_kms)
    wanted = set(reference["object_id"].astype("int64"))
    reference_rows = (
        {int(r["object_id"]): r for _, r in reference.iterrows()}
        if include_catalogue_hypotheses
        else {}
    )
    rows: list[dict] = []
    for path in files:
        with open_sir_file(str(path)) as sir:
            for group in sir.groups().values():
                if group.object_id not in wanted:
                    continue
                spectrum = sir.read_combined(group)
                metrics = spectrum.quality_metrics()
                if metrics["usable_pixel_fraction"] < min_usable_fraction:
                    continue
                per_object = hypotheses
                if include_catalogue_hypotheses:
                    per_object = deduplicate(
                        [*hypotheses, *from_catalogue(reference_rows[group.object_id])]
                    )
                best = blind_best_redshift(spectrum, settings, hypotheses=per_object)
                if best is None:
                    continue
                best.update(
                    {
                        "object_id": group.object_id,
                        "tile_id": sir.tile_id,
                        "lsf_sigma": spectrum.lsf_sigma,
                        "usable_pixel_fraction": metrics["usable_pixel_fraction"],
                        "median_snr_per_pixel": metrics["median_snr_per_pixel"],
                    }
                )
                rows.append(best)
                if max_objects is not None and len(rows) >= max_objects:
                    break
        if max_objects is not None and len(rows) >= max_objects:
            break

    table = pd.DataFrame(rows)
    if table.empty:
        return BlindRedshiftResult(table=table, compared=pd.DataFrame())
    compared = compare_redshifts(
        table, reference, reference_column=reference_column, tolerance_kms=tolerance_kms
    )
    for column, bins, labels in (
        (
            "spe_best_snr",
            [0, 5, 10, 20, np.inf],
            ["SNR 3-5", "SNR 5-10", "SNR 10-20", "SNR > 20"],
        ),
    ):
        if column in compared:
            compared["snr_bin"] = pd.cut(compared[column], bins=bins, labels=labels)
    return BlindRedshiftResult(table=table, compared=compared)


def catastrophic_fraction(compared: pd.DataFrame, tolerance_kms: float = 1000.0) -> float:
    """Fraction of comparisons that disagree by more than the tolerance."""
    if compared.empty:
        return float("nan")
    return float(np.mean(np.abs(compared["delta_v_kms"]) >= tolerance_kms))
