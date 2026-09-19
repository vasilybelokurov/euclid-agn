"""Broad-line decomposition on quasars that are known to have broad lines.

Every earlier calibration of the AGN detection statistic rested on injections
into Q1 spectra and on empirical nulls.  This measures the same statistic on
**confirmed AGN**: DESI DR1 quasars with Euclid spectra
(`validation.qso_truth`), with the redshift held at the DESI value so that the
decomposition is judged on its own and not on our redshift.

Configuration follows what the rest of the project has learnt:

* the nuisance continuum is the **power law**, not the spline - a 12-knot
  spline recovers 306 % of a 6000 km/s line's flux and halves its Delta
  chi-squared (`tests/unit/test_continuum_choice.py`);
* the LSF is the value fitted per object, not the archive ``LSF_SIG``;
* the variance is rescaled by the dither-to-dither scatter, and spectra
  carrying the decontamination trough are flagged.

For each object the broad width is scanned over ``sigma_grid`` and the best
Delta chi-squared kept, with the containment and edge guards that stop a
"broad line" being fitted to the end of the spectrum.

Usage::

    python -m euclid_agn.validation.agn_measurement --out outputs/agn_on_known_qsos.parquet
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from euclid_agn.models.broad import BroadComponent, BroadFamily
from euclid_agn.models.forward import fit_hypothesis
from euclid_agn.models.line_catalog import BY_NAME, SYSTEMS
from euclid_agn.models.narrow import NarrowSystem, velocity_grid
from euclid_agn.io.sir import open_sir_file
from euclid_agn.spectra.artefacts import continuum_trough
from euclid_agn.spectra.coherence import dither_variance_rescale

log = logging.getLogger(__name__)

#: Broad widths scanned, km/s.  Type-1 AGN span FWHM 1000-20000 km/s, i.e. sigma 400-8500.
SIGMA_GRID: tuple[float, ...] = (750.0, 1200.0, 1800.0, 2500.0, 3500.0, 5000.0, 7000.0, 9000.0)
WAVELENGTH_MIN, WAVELENGTH_MAX = 12500.0, 18500.0


def visible_permitted(z: float, wavelength: np.ndarray, edge_margin: float = 200.0) -> list[str]:
    """Permitted lines inside the usable range at this redshift, brightest first."""
    lo, hi = max(wavelength.min(), WAVELENGTH_MIN) + edge_margin, min(wavelength.max(), WAVELENGTH_MAX) - edge_margin
    order = ("Halpha", "Hbeta", "Pabeta", "Pagamma", "HeI10830", "Hgamma", "Hdelta", "MgII2799")
    return [n for n in order if n in BY_NAME and lo <= BY_NAME[n].rest * (1 + z) <= hi]


def narrow_system_for(z: float, lsf_sigma: float, broad_line: str) -> NarrowSystem:
    """The narrow system containing ``broad_line``, so the broad fit has its narrow counterpart."""
    members = next((s.members for s in SYSTEMS if broad_line in s.members), (broad_line,))
    # keep only the lines actually inside the grism at this redshift, else the narrow block
    # carries columns of zeros
    visible = tuple(n for n in members
                    if WAVELENGTH_MIN < BY_NAME[n].rest * (1 + z) < WAVELENGTH_MAX) or (broad_line,)
    return NarrowSystem(visible, z=z, lsf_sigma=lsf_sigma, velocities=velocity_grid(600.0, 200.0))


def broad_orthogonality(column: np.ndarray, continuum_design: np.ndarray, weight: np.ndarray) -> float:
    """Fraction of a broad column the continuum block cannot absorb.

    A sigma = 8000 km/s Gaussian spans ~900 A at 1.4 um, which is the scale on
    which any continuum model curves, so the widest components are partly
    degenerate with the continuum and will happily soak up continuum mismatch.
    The screening stage has guarded against this since session 5
    (``min_broad_orthogonality``); the AGN measurement must too.  The
    threshold is calibrated on the confirmed quasars: at 0.5, 34 % of detected
    widths rail at the top of the scan (FWHM 21,000 km/s, unphysical); at 0.8
    none do and 90 % of the detections survive; at 0.9 a third are lost.
    """
    w = np.asarray(column, dtype=np.float64) * weight
    norm = float(np.linalg.norm(w))
    if norm <= 0:
        return 0.0
    basis, _ = np.linalg.qr(continuum_design * weight[:, None])
    residual = w - basis @ (basis.T @ w)
    return float(np.linalg.norm(residual) / norm)


def measure(spectrum, z: float, lsf_sigma: float, sigma_grid=SIGMA_GRID, continuum: str = "power_law",
            min_containment: float = 0.8, min_orthogonality: float = 0.8) -> dict:
    """Best broad-line detection for one spectrum at a fixed redshift."""
    ok = spectrum.usable() & (spectrum.wavelength >= WAVELENGTH_MIN) & (spectrum.wavelength <= WAVELENGTH_MAX)
    if ok.sum() < 100:
        return {}
    w, f, v = spectrum.wavelength[ok], spectrum.flux[ok], spectrum.variance[ok]
    lines = visible_permitted(z, w)
    if not lines:
        return {"n_permitted_visible": 0}
    from euclid_agn.models.forward import continuum_for

    continuum_design, _, _ = continuum_for(w, continuum)
    weight = 1.0 / np.sqrt(v)
    best = {"n_permitted_visible": len(lines), "broad_delta_chi2": -np.inf, "n_widths_rejected": 0}
    for line in lines:
        narrow = narrow_system_for(z, lsf_sigma, line)
        for sigma in sigma_grid:
            component = BroadComponent(line_name=line, z=z, sigma_kms=sigma, lsf_sigma=lsf_sigma)
            if component.contained_fraction(w, spectrum.bin_width) < min_containment:
                best["n_widths_rejected"] += 1
                continue
            orthogonality = broad_orthogonality(component.basis(w, spectrum.bin_width), continuum_design, weight)
            if orthogonality < min_orthogonality:
                best["n_widths_rejected"] += 1
                continue
            family = BroadFamily(line_names=(line,), z=z, sigma_kms=sigma, lsf_sigma=lsf_sigma)
            fit = fit_hypothesis(w, f, v, narrow, family, bin_width=spectrum.bin_width, continuum=continuum)
            if fit.delta_chi2 > best["broad_delta_chi2"]:
                flux = fit.broad_fluxes.get(line, np.nan)
                error = fit.broad_flux_errors.get(line, np.nan)
                best.update({"broad_delta_chi2": float(fit.delta_chi2), "broad_line": line,
                             "broad_sigma_kms": float(sigma), "broad_fwhm_kms": float(component.fwhm_kms),
                             "broad_flux": float(flux), "broad_flux_err": float(error),
                             "broad_snr": float(flux / error) if error and np.isfinite(error) and error > 0 else np.nan,
                             "broad_containment": float(component.contained_fraction(w, spectrum.bin_width)),
                             "broad_orthogonality": float(orthogonality),
                             "narrow_flux": float(fit.narrow_fluxes.get(line, np.nan))})
    return best


def run_store(store: Path, truth: pd.DataFrame, continuum: str = "power_law", lsf: float = 13.7,
              limit: int | None = None) -> pd.DataFrame:
    """Measure every spectrum in an extracted store (no archive access).

    The store carries the combined spectrum only, so the per-object dither
    variance rescaling is unavailable and a global inflation was applied at
    extraction; ``lsf`` is the smoothing width used for the line profiles,
    defaulting to NISP's point-source value rather than the unreliable header.
    """
    from euclid_agn.validation.qso_extract import load_store

    meta, spectrum_at = load_store(store)
    truth = truth.drop_duplicates("object_id").set_index("object_id")
    rows, started = [], time.time()
    for i in range(len(meta) if limit is None else min(limit, len(meta))):
        row = meta.iloc[i]
        oid = int(row.object_id)
        if oid not in truth.index:
            continue
        spectrum = spectrum_at(i)
        z = float(truth.loc[oid, "desi_z"])
        out = {"object_id": oid, "desi_z": z, "snr": float(row.snr), "lsf_used": lsf,
               "usable_fraction": float(row.usable_fraction), "continuum": continuum,
               "H_AB": float(truth.loc[oid].get("H_AB", np.nan))}
        out.update(measure(spectrum, z, lsf, continuum=continuum))
        rows.append(out)
        if len(rows) % 250 == 0:
            rate = len(rows) / (time.time() - started)
            log.info("%d measured, %.1f/s, %.0f min left", len(rows), rate, (len(meta) - len(rows)) / max(rate, 1e-9) / 60)
    log.info("%d objects in %.0f s", len(rows), time.time() - started)
    return pd.DataFrame(rows)


def run(sample: pd.DataFrame, continuum: str = "power_law") -> pd.DataFrame:
    rows = []
    started = time.time()
    for path, group in sample.groupby("file"):
        with open_sir_file(path) as handle:
            for _, row in group.iterrows():
                try:
                    obs = handle.read_observation(int(row.object_id), with_dithers=True)
                except KeyError:
                    continue
                spectrum, _ = dither_variance_rescale(obs) if obs.dithers else (obs.combined, None)
                lsf = float(row.get("lsf_fitted", np.nan))
                if not np.isfinite(lsf) or lsf <= 0:
                    lsf = float(spectrum.lsf_sigma)
                ok = spectrum.usable()
                out = {"object_id": int(row.object_id), "desi_z": float(row.desi_z), "lsf_used": lsf,
                       "snr": float(np.nanmedian(spectrum.flux[ok] / np.sqrt(spectrum.variance[ok]))) if ok.any() else np.nan,
                       "trough": continuum_trough(spectrum).flagged, "continuum": continuum}
                out.update(measure(spectrum, float(row.desi_z), lsf, continuum=continuum))
                rows.append(out)
    log.info("%d objects in %.0f s", len(rows), time.time() - started)
    return pd.DataFrame(rows)


def summarise(t: pd.DataFrame, thresholds=(25, 50, 100)) -> str:
    d = t[np.isfinite(t.get("broad_delta_chi2", pd.Series(dtype=float)))]
    lines = [f"n = {len(t)} confirmed quasars, {len(d)} with a broad line fittable in range"]
    for cut in thresholds:
        lines.append(f"  broad Delta chi2 > {cut:3d}: {(d.broad_delta_chi2 > cut).sum():3d} ({(d.broad_delta_chi2 > cut).mean():.0%})")
    if "desi_z" in d:
        for lo, hi in ((0, 0.9), (0.9, 1.5), (1.5, 2.5), (2.5, 6)):
            s = d[(d.desi_z >= lo) & (d.desi_z < hi)]
            if len(s):
                lines.append(f"    z {lo}-{hi}: n={len(s):4d}, detected (>25) {(s.broad_delta_chi2 > 25).mean():.0%}")
    for lo, hi in ((0, 3), (3, 10), (10, 1e9)):
        s = d[(d.snr >= lo) & (d.snr < hi)]
        if len(s):
            lines.append(f"    S/N {lo}-{hi}: n={len(s):3d}, detected (>25) {(s.broad_delta_chi2 > 25).mean():.0%}, "
                         f"median FWHM {s[s.broad_delta_chi2 > 25].broad_fwhm_kms.median():.0f} km/s")
    if "trough" in d:
        clean = d[~d.trough]
        lines.append(f"  excluding trough-flagged: n={len(clean)}, detected {(clean.broad_delta_chi2 > 25).mean():.0%}")
    det = d[d.broad_delta_chi2 > 25]
    if len(det):
        lines.append(f"  detected: median flux {det.broad_flux.median() * 1e17:.1f}e-17, median broad S/N {det.broad_snr.median():.1f}, "
                     f"FWHM quartiles {np.round(det.broad_fwhm_kms.quantile([.25, .5, .75])).astype(int).tolist()} km/s")
        lines.append(f"  FWHM at the scan edge (railed): {(det.broad_sigma_kms >= max(SIGMA_GRID)).mean():.0%}")
    if "broad_line" in d:
        lines.append(f"  line carrying the detection: {d[d.broad_delta_chi2 > 25].broad_line.value_counts().to_dict()}")
    return "\n".join(lines)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sample", type=Path, default=Path("outputs/qso_fits_with_spe.parquet"))
    parser.add_argument("--truth", type=Path, default=Path("outputs/qso_truth_matched.parquet"))
    parser.add_argument("--continuum", default="power_law")
    parser.add_argument("--store", type=Path, default=None, help="extracted spectra store (skips the archive)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=Path, default=Path("outputs/agn_on_known_qsos.parquet"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.store:
        truth = pd.read_parquet(args.truth)
        print(f"store {args.store}; continuum = {args.continuum}")
        t = run_store(args.store, truth, continuum=args.continuum, limit=args.limit)
    else:
        sample = pd.read_parquet(args.sample)
        truth = pd.read_parquet(args.truth)[["object_id", "file"]]
        sample = sample.merge(truth.drop_duplicates("object_id"), on="object_id", how="left").drop_duplicates("object_id")
        print(f"{len(sample)} confirmed quasars; continuum = {args.continuum}")
        t = run(sample, continuum=args.continuum)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    t.to_parquet(args.out, index=False)
    print(summarise(t))


if __name__ == "__main__":
    main()
