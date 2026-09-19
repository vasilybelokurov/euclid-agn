"""Gallery of the broad-line decomposition, split into detections and non-detections.

Two subfolders, because the interesting failures are as informative as the
successes:

* ``detected/``     - Delta chi2 above the threshold, sorted by Delta chi2, so
  the strongest AGN signatures come first and the marginal ones last;
* ``not_detected/`` - sorted by S/N *descending*, so the first pages are the
  bright spectra where a confirmed quasar's broad line was **not** recovered.
  Those are the ones worth understanding.

Each panel shows the Euclid spectrum with its +-1 sigma band, the full model
(continuum + narrow + broad), the continuum alone and the broad component
alone, so it is visible whether the fit is describing a line or reshaping the
continuum.

Usage::

    python -m euclid_agn.validation.agn_gallery --out-dir plots/agn_gallery
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from euclid_agn.models.broad import BroadFamily
from euclid_agn.models.forward import fit_hypothesis
from euclid_agn.validation.agn_measurement import WAVELENGTH_MAX, WAVELENGTH_MIN, narrow_system_for
from euclid_agn.validation.qso_extract import load_store

log = logging.getLogger(__name__)


def refit(spectrum, row, continuum: str = "power_law"):
    """Re-run the winning hypothesis and return the pieces needed to draw it."""
    ok = spectrum.usable() & (spectrum.wavelength >= WAVELENGTH_MIN) & (spectrum.wavelength <= WAVELENGTH_MAX)
    if ok.sum() < 100 or not isinstance(row.broad_line, str):
        return None
    w, f, v = spectrum.wavelength[ok], spectrum.flux[ok], spectrum.variance[ok]
    z, lsf, sigma = float(row.desi_z), float(row.lsf_used), float(row.broad_sigma_kms)
    narrow = narrow_system_for(z, lsf, row.broad_line)
    family = BroadFamily(line_names=(row.broad_line,), z=z, sigma_kms=sigma, lsf_sigma=lsf)
    fit = fit_hypothesis(w, f, v, narrow, family, bin_width=spectrum.bin_width, continuum=continuum)
    if fit.m1 is None or fit.blocks_m1 is None:
        return None
    design, coefficients = fit.blocks_m1.design, fit.m1.coefficients
    piece = {}
    for name in ("continuum", "narrow", "broad"):
        sl = fit.blocks_m1.slices.get(name)
        piece[name] = design[:, sl] @ coefficients[sl] if sl is not None else np.zeros(w.size)
    return w, f, np.sqrt(v), fit.m1.model, piece


def panel(ax, drawn, row):
    w, flux, sigma, model, piece = drawn
    scale = 1e17
    ax.fill_between(w, (flux - sigma) * scale, (flux + sigma) * scale, color="0.8", lw=0)
    ax.plot(w, flux * scale, color="k", lw=0.55)
    ax.plot(w, piece["continuum"] * scale, color="tab:blue", lw=0.8, ls="--")
    ax.plot(w, (piece["continuum"] + piece["broad"]) * scale, color="tab:orange", lw=0.8)
    ax.plot(w, model * scale, color="crimson", lw=1.0)
    centre = np.average(w, weights=np.maximum(piece["broad"], 0)) if piece["broad"].max() > 0 else np.nan
    if np.isfinite(centre):
        ax.axvline(centre, color="tab:orange", lw=0.6, alpha=0.5)
    ax.set_title(f"{int(row.object_id)}  z={row.desi_z:.3f}  S/N {row.snr:.1f}\n"
                 f"{row.broad_line} FWHM {row.broad_fwhm_kms:.0f} km/s, Δχ²={row.broad_delta_chi2:.0f}, "
                 f"flux {row.broad_flux * 1e17:.0f}e-17 ({row.broad_snr:.1f}σ), orth {row.broad_orthogonality:.2f}",
                 fontsize=6.0)
    ax.tick_params(labelsize=6)


def build(rows: pd.DataFrame, spectrum_at, positions: dict, out_dir: Path, title: str, per_page: int = 12,
          max_png: int = 6, dpi: int = 110) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    out_dir.mkdir(parents=True, exist_ok=True)
    drawn = []
    for _, row in rows.iterrows():
        i = positions.get(int(row.object_id))
        if i is None:
            continue
        got = refit(spectrum_at(i), row)
        if got is not None:
            drawn.append((got, row))
    ncols, nrows = 3, int(np.ceil(per_page / 3))
    n_pages = int(np.ceil(len(drawn) / per_page))
    with PdfPages(out_dir / "gallery.pdf") as pdf:
        for page in range(n_pages):
            chunk = drawn[page * per_page:(page + 1) * per_page]
            fig, axes = plt.subplots(nrows, ncols, figsize=(14, 2.6 * nrows))
            for ax in np.atleast_1d(axes).ravel():
                ax.axis("off")
            for ax, (got, row) in zip(np.atleast_1d(axes).ravel(), chunk):
                ax.axis("on")
                panel(ax, got, row)
            fig.supxlabel("observed wavelength [Å]", fontsize=8)
            fig.supylabel(r"$F_\lambda$ [$10^{-17}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$]", fontsize=8)
            fig.suptitle(f"{title} — page {page + 1}/{n_pages}.  data black, model red, continuum blue dashed, "
                         "continuum+broad orange", fontsize=9)
            fig.tight_layout(rect=(0.01, 0.01, 1, 0.95))
            pdf.savefig(fig)
            if page < max_png:
                fig.savefig(out_dir / f"page{page + 1:02d}.png", dpi=dpi)
            plt.close(fig)
    rows.head(len(drawn)).to_parquet(out_dir / "index.parquet", index=False)
    return len(drawn)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results", type=Path, default=Path("outputs/agn_on_qsos_edfn.parquet"))
    parser.add_argument("--store", type=Path, default=Path("outputs/qso_spectra_edfn.npz"))
    parser.add_argument("--out-dir", type=Path, default=Path("plots/agn_gallery"))
    parser.add_argument("--threshold", type=float, default=25.0)
    parser.add_argument("--max-each", type=int, default=240)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    results = pd.read_parquet(args.results)
    results = results[results.broad_delta_chi2.notna() & results.broad_line.notna()]
    meta, spectrum_at = load_store(args.store)
    positions = {int(o): i for i, o in enumerate(meta.object_id)}
    started = time.time()
    worked = results[results.broad_delta_chi2 > args.threshold].sort_values("broad_delta_chi2", ascending=False).head(args.max_each)
    failed = results[results.broad_delta_chi2 <= args.threshold].sort_values("snr", ascending=False).head(args.max_each)
    n_w = build(worked, spectrum_at, positions, args.out_dir / "detected",
                f"Broad line DETECTED (Δχ² > {args.threshold:.0f}), strongest first")
    n_f = build(failed, spectrum_at, positions, args.out_dir / "not_detected",
                f"Broad line NOT detected (Δχ² ≤ {args.threshold:.0f}) in a confirmed quasar, brightest first")
    print(f"detected/     {n_w} panels  (of {int((results.broad_delta_chi2 > args.threshold).sum())} detections)")
    print(f"not_detected/ {n_f} panels  (of {int((results.broad_delta_chi2 <= args.threshold).sum())} non-detections)")
    print(f"{time.time() - started:.0f} s -> {args.out_dir}")


if __name__ == "__main__":
    main()
