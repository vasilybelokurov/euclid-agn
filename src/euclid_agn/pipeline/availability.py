"""How many Euclid spectra exist, and how many are actually usable.

Two different questions, answered separately because they have very different
answers.

*Availability* is a catalogue question: how many sources have an extracted
spectrum at all.  It is answered with counts from the association table.

*Usability* is a pixel-level question: of the spectra that exist, how many have
enough unmasked pixels inside the science window to be worth fitting.  It can
only be answered by opening files, so it is measured on a tile sample and
reported as a distribution, never as a single number.

Both feed the parent-sample definition.  The project's claim is host-unbiased
selection *within the available spectroscopic parent sample*, so the parent
sample has to be characterised before anything is claimed about it.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd

from euclid_agn.archive import schema
from euclid_agn.archive.irsa import IrsaQ1Backend
from euclid_agn.constants import RGS_SCIENCE_WMAX_ANGSTROM, RGS_SCIENCE_WMIN_ANGSTROM
from euclid_agn.io.sir import open_sir_file

log = logging.getLogger(__name__)


def count_availability(
    backend: IrsaQ1Backend, fields: Sequence[str] | None = None
) -> pd.DataFrame:
    """One row per field: tiles, association rows, rows with a spectrum.

    An association row exists for every MER source in a tile that the
    spectroscopic pipeline considered; ``path IS NOT NULL`` marks the ones for
    which a spectrum was actually extracted.
    """
    fields = list(fields or schema.FIELDS)
    rows = []
    for field in fields:
        tiles = backend.list_tiles(field)
        tile_clause = backend._tile_clause("tileid", tiles)  # noqa: SLF001
        total = backend.tap.query(
            f"SELECT COUNT(*) AS n FROM {schema.SPECTRA_ASSOCIATION} WHERE {tile_clause}"
        )["n"].iloc[0]
        with_spectrum = backend.tap.query(
            f"SELECT COUNT(*) AS n FROM {schema.SPECTRA_ASSOCIATION} "
            f"WHERE path IS NOT NULL AND {tile_clause}"
        )["n"].iloc[0]
        rows.append(
            {
                "field": field,
                "n_tiles": len(tiles),
                "n_association_rows": int(total),
                "n_with_spectrum": int(with_spectrum),
                "spectrum_fraction": float(with_spectrum) / float(total) if total else np.nan,
            }
        )
        log.info("%s: %d tiles, %d/%d with a spectrum", field, len(tiles), with_spectrum, total)
    return pd.DataFrame(rows)


def spectrum_quality_table(
    path: str,
    filesystem=None,
    max_objects: int | None = None,
    wavelength_min: float = RGS_SCIENCE_WMIN_ANGSTROM,
    wavelength_max: float = RGS_SCIENCE_WMAX_ANGSTROM,
) -> pd.DataFrame:
    """Per-object usability metrics for every object in one SIR file.

    ``usable_fraction_science`` is the fraction of pixels inside
    ``[wavelength_min, wavelength_max]`` that survive masking; it is the number
    that matters, because the pixels outside that interval are not used.
    """
    rows = []
    with open_sir_file(path, filesystem=filesystem) as sir:
        groups = list(sir.groups().values())
        if max_objects is not None:
            groups = groups[:max_objects]
        for group in groups:
            combined = sir.read_combined(group)
            metrics = combined.quality_metrics()
            in_window = (combined.wavelength >= wavelength_min) & (
                combined.wavelength <= wavelength_max
            )
            usable = combined.usable()
            n_window = int(np.count_nonzero(in_window))
            n_usable_window = int(np.count_nonzero(in_window & usable))
            with np.errstate(invalid="ignore", divide="ignore"):
                snr = combined.flux / np.sqrt(combined.variance)
            good = in_window & usable
            rows.append(
                {
                    "object_id": group.object_id,
                    "tile_id": sir.tile_id,
                    "n_dither_hdus": group.n_dither_hdus,
                    "lsf_sigma": combined.lsf_sigma,
                    "usable_fraction_all": metrics["usable_pixel_fraction"],
                    "usable_fraction_science": (
                        n_usable_window / n_window if n_window else np.nan
                    ),
                    "n_usable_science": n_usable_window,
                    "median_quality": metrics["median_quality"],
                    "mask_frac_not_use": metrics["mask_frac_not_use"],
                    "mask_frac_low_snr": metrics["mask_frac_low_snr"],
                    "median_snr_science": (
                        float(np.median(snr[good])) if np.any(good) else np.nan
                    ),
                    "ndith_max": metrics.get("ndith_max", np.nan),
                }
            )
    return pd.DataFrame(rows)


def summarise_quality(table: pd.DataFrame, min_usable_fraction: float = 0.5) -> dict[str, float]:
    """Headline usability numbers from a quality table."""
    n = len(table)
    if n == 0:
        return {"n_objects": 0}
    usable = table["usable_fraction_science"]
    percentiles = np.nanpercentile(usable, [5, 25, 50, 75, 95])
    lsf = np.nanpercentile(table["lsf_sigma"], [50, 90, 99])
    return {
        "n_objects": n,
        "frac_above_threshold": float(np.mean(usable >= min_usable_fraction)),
        "usable_fraction_p05": float(percentiles[0]),
        "usable_fraction_p25": float(percentiles[1]),
        "usable_fraction_median": float(percentiles[2]),
        "usable_fraction_p75": float(percentiles[3]),
        "usable_fraction_p95": float(percentiles[4]),
        "frac_completely_masked": float(np.mean(usable <= 0.0)),
        "lsf_sigma_median": float(lsf[0]),
        "lsf_sigma_p90": float(lsf[1]),
        "lsf_sigma_p99": float(lsf[2]),
        "median_snr_science": float(np.nanmedian(table["median_snr_science"])),
    }
