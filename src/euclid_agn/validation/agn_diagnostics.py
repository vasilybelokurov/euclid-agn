"""Two failure modes of the broad-line measurement, found automatically.

Reading the gallery shows both, so they should be measurable rather than
spotted by eye:

**Spurious detections.**  A genuine broad line spreads its signal over tens of
pixels - sigma = 3000 km/s is 11 pixels, so the line covers ~25 - while a
cosmic ray or a decontamination edge lives in one to three.  The fraction of
the total Delta chi-squared contributed by the three best pixels therefore
separates them: near 1 for a spike, small for a line.

**Missed lines.**  If the data hold an emission line the model did not fit,
the residual ``data - model`` carries a broad positive excursion.  Smoothing
the residual on the scale of a broad line and taking its largest significance
finds those, whether or not the line sits where a permitted transition is
expected (it may be a transition we do not have, or the redshift may be wrong).

Usage::

    python -m euclid_agn.validation.agn_diagnostics --out outputs/agn_diagnostics.parquet
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from euclid_agn.models.broad import BroadFamily
from euclid_agn.models.forward import fit_hypothesis
from euclid_agn.validation.agn_measurement import WAVELENGTH_MAX, WAVELENGTH_MIN, narrow_system_for
from euclid_agn.validation.qso_extract import load_store

log = logging.getLogger(__name__)


def diagnose(spectrum, row, continuum: str = "power_law", smooth_pixels: int = 11) -> dict:
    """Concentration of the detection and the largest unmodelled emission residual."""
    ok = spectrum.usable() & (spectrum.wavelength >= WAVELENGTH_MIN) & (spectrum.wavelength <= WAVELENGTH_MAX)
    if ok.sum() < 100 or not isinstance(row.broad_line, str):
        return {}
    w, f, v = spectrum.wavelength[ok], spectrum.flux[ok], spectrum.variance[ok]
    z, lsf, sigma = float(row.desi_z), float(row.lsf_used), float(row.broad_sigma_kms)
    narrow = narrow_system_for(z, lsf, row.broad_line)
    family = BroadFamily(line_names=(row.broad_line,), z=z, sigma_kms=sigma, lsf_sigma=lsf)
    fit = fit_hypothesis(w, f, v, narrow, family, bin_width=spectrum.bin_width, continuum=continuum)
    if fit.m1 is None:
        return {}
    r0 = (f - fit.m0.model) ** 2 / v
    r1 = (f - fit.m1.model) ** 2 / v
    per_pixel = r0 - r1                      # each pixel's contribution to the detection
    total = float(per_pixel.sum())
    top3 = float(np.sort(per_pixel)[-3:].sum())
    # unmodelled emission: residual above the full model, smoothed on a broad-line scale
    kernel = np.ones(smooth_pixels) / np.sqrt(smooth_pixels)
    residual = np.convolve((f - fit.m1.model) / np.sqrt(v), kernel, mode="same")
    peak = int(np.argmax(residual))
    return {"chi2_concentration": top3 / total if total > 0 else np.nan,
            "n_pixels_90pc": int(np.searchsorted(np.cumsum(np.sort(per_pixel)[::-1]), 0.9 * total) + 1) if total > 0 else -1,
            "residual_peak_sigma": float(residual[peak]), "residual_peak_wavelength": float(w[peak]),
            "residual_peak_z_halpha": float(w[peak] / 6564.6 - 1), "model_chi2_reduced": float(r1.sum() / max(ok.sum() - 6, 1))}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results", type=Path, default=Path("outputs/agn_on_qsos_edfn.parquet"))
    parser.add_argument("--store", type=Path, default=Path("outputs/qso_spectra_edfn.npz"))
    parser.add_argument("--out", type=Path, default=Path("outputs/agn_diagnostics.parquet"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    results = pd.read_parquet(args.results)
    results = results[results.broad_delta_chi2.notna() & results.broad_line.notna()]
    if args.limit:
        results = results.head(args.limit)
    meta, spectrum_at = load_store(args.store)
    positions = {int(o): i for i, o in enumerate(meta.object_id)}
    rows = []
    for _, row in results.iterrows():
        i = positions.get(int(row.object_id))
        if i is None:
            continue
        d = diagnose(spectrum_at(i), row)
        if d:
            rows.append({"object_id": int(row.object_id), "desi_z": row.desi_z, "snr": row.snr,
                         "broad_delta_chi2": row.broad_delta_chi2, "broad_line": row.broad_line,
                         "broad_fwhm_kms": row.broad_fwhm_kms, **d})
    t = pd.DataFrame(rows)
    t.to_parquet(args.out, index=False)
    det = t[t.broad_delta_chi2 > 25]
    miss = t[t.broad_delta_chi2 <= 25]
    print(f"{len(t)} diagnosed: {len(det)} detections, {len(miss)} non-detections")
    print(f"\nSPURIOUS DETECTIONS - Delta chi2 concentrated in a few pixels:")
    for cut in (0.5, 0.7, 0.9):
        s = det[det.chi2_concentration > cut]
        print(f"  top-3 pixels carry > {cut:.0%} of the signal: {len(s):4d} ({len(s)/max(len(det),1):.0%} of detections)")
    print(f"  median pixels holding 90 % of the signal: real-looking {det[det.chi2_concentration<0.5].n_pixels_90pc.median():.0f}"
          f" vs spike-like {det[det.chi2_concentration>0.7].n_pixels_90pc.median():.0f}")
    print(f"\nMISSED LINES - unmodelled emission in non-detections:")
    for cut in (5, 8, 12):
        s = miss[miss.residual_peak_sigma > cut]
        print(f"  residual peak > {cut:2d} sigma: {len(s):4d} ({len(s)/max(len(miss),1):.0%} of non-detections)")
    worst = miss.nlargest(8, "residual_peak_sigma")
    print("\n  strongest unmodelled emission (object, z, S/N, peak sigma, peak wavelength, z if it were H-alpha):")
    for _, r in worst.iterrows():
        print(f"    {int(r.object_id)} z={r.desi_z:.3f} S/N={r.snr:5.1f} peak={r.residual_peak_sigma:5.1f}sigma "
              f"at {r.residual_peak_wavelength:.0f} A (H-alpha z would be {r.residual_peak_z_halpha:.3f})")


if __name__ == "__main__":
    main()
