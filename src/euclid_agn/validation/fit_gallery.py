"""A gallery of every measured spectrum with the model we fitted to it.

One panel per object: the extracted spectrum with its ±1σ band, the best-fit
model drawn on top, and the numbers that decide whether to believe it — S/N,
LSF width (which says resolved or not), the winning class, the redshift or
velocity, the Δχ² margin and the reduced χ².  Objects are sorted by S/N so the
first pages carry the spectra that matter and the last pages show what the
faint end really looks like.

Both classes are scanned for every object with the same machinery used in the
analysis — GALAXY archetypes over 0 < z < 1 and PHOENIX stars over ±600 km/s,
non-negative, multiplicative continuum polynomial, dither-scatter variance —
and the winner by χ² is the model drawn.  The model is re-solved at the best
redshift by :func:`euclid_agn.fit.template_cube.best_fit_model`, which
reproduces the scan's χ² to 2 %.

Usage::

    python -m euclid_agn.validation.fit_gallery --candidates outputs/gc_candidates_ngc1527.parquet \
        --out-dir plots/gallery_ngc1527 --min-snr 10 --max-objects 700
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from euclid_agn.constants import C_KMS
from euclid_agn.fit.screen import ScreenSettings, prepare
from euclid_agn.fit.template_cube import CubeStore, best_fit_model, cube_scan, redshift_grid
from euclid_agn.io.sir import open_sir_file
from euclid_agn.models.library import load_phoenix_library, load_xsl_ssp_library
from euclid_agn.spectra.coherence import dither_variance_rescale

log = logging.getLogger(__name__)


def build_stores(wavelength, bin_width, z_max: float = 1.0, step_kms: float = 300.0, lsf_step: float = 10.0):
    galaxy = load_xsl_ssp_library(log_age_min=8.5, mh_min=-0.5)[::18]
    stars = [t for t in load_phoenix_library(mh_values=(0.0, -1.0), teff_step=5) if t.metadata["logg"] in (2.0, 4.5)]
    gal = CubeStore({"GALAXY": galaxy}, redshift_grid(0.0, z_max, step_kms), wavelength, bin_width, lsf_step=lsf_step)
    star = CubeStore({"STAR": stars}, np.linspace(-0.002, 0.002, 9), wavelength, bin_width, lsf_step=lsf_step)
    return gal, star


def fit_one(spectrum, projected, gal_store, star_store):
    """Scan both classes; return (kind, result, cube) for the better chi-squared."""
    out = {}
    for kind, store in (("GALAXY", gal_store), ("STAR", star_store)):
        cube = store.get(kind, spectrum.lsf_sigma)
        res = cube_scan(spectrum, projected, cube, poly_degree=0, nonnegative=True, multiplicative_degree=3)
        if res is not None:
            out[kind] = (res, cube)
    if not out:
        return None, None, None
    kind = min(out, key=lambda k: out[k][0].chi2)
    return kind, out[kind][0], out[kind][1]


def panel(ax, spectrum, projected, model, row, kind, result, host_v=np.nan):
    keep = np.isin(spectrum.wavelength, projected.wavelength)
    w = projected.wavelength
    flux = spectrum.flux[keep] * 1e17
    sigma = np.sqrt(spectrum.variance[keep]) * 1e17
    ax.fill_between(w, flux - sigma, flux + sigma, color="0.78", lw=0)
    ax.plot(w, flux, color="k", lw=0.55)
    ax.plot(w, model * 1e17, color="crimson", lw=0.95)
    v = C_KMS * result.z
    label = f"v = {v:+.0f} km/s" if kind == "STAR" or abs(result.z) < 0.02 else f"z = {result.z:.3f}"
    red = result.chi2 / max(result.n_pixels - result.n_parameters, 1)
    sep = f", {row.sep_arcmin:.1f}′" if np.isfinite(row.get("sep_arcmin", np.nan)) else ""
    # a reduced chi-squared far above one on a smooth continuum means the *data* are not describable
    # by any stellar population - in this field that is order overlap, detector edges and cliffs
    bad = red > 20
    tag = "  ✗ unfittable" if bad else ""
    ax.set_title(f"{int(row.object_id)}{sep}\nS/N {row.snr:.0f}, LSF {row.lsf_sigma:.0f} Å, {kind}, {label}, "
                 f"Δχ²={result.delta_chi2_runner_up:.0f}, χ²ᵣ={red:.1f}{tag}", fontsize=6.2,
                 color="tab:red" if bad else "black")
    ax.tick_params(labelsize=6)
    if np.isfinite(host_v) and abs(v - host_v) < 500 and kind != "STAR":
        for spine in ax.spines.values():
            spine.set_edgecolor("tab:blue"); spine.set_linewidth(1.8)


def run(candidates: pd.DataFrame, out_dir: Path, per_page: int = 12, host_v: float = np.nan, dpi: int = 110) -> pd.DataFrame:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    out_dir.mkdir(parents=True, exist_ok=True)
    gal_store = star_store = None
    rows, panels = [], []
    started = time.time()
    for path, group in candidates.groupby("file"):
        with open_sir_file(path) as f:
            for _, row in group.iterrows():
                try:
                    obs = f.read_observation(int(row.object_id), with_dithers=True)
                except KeyError:
                    continue
                spectrum, _ = dither_variance_rescale(obs) if obs.dithers else (obs.combined, None)
                projected = prepare(spectrum, ScreenSettings(n_knots=1, outlier_threshold=5.0))
                if projected is None:
                    continue
                if gal_store is None:
                    gal_store, star_store = build_stores(spectrum.wavelength, spectrum.bin_width)
                kind, result, cube = fit_one(spectrum, projected, gal_store, star_store)
                if result is None:
                    continue
                _, model = best_fit_model(spectrum, projected, cube, result.z, poly_degree=0,
                                          nonnegative=True, multiplicative_degree=3)
                panels.append((spectrum, projected, model, row, kind, result))
                rows.append({"object_id": int(row.object_id), "snr": float(row.snr), "lsf_sigma": float(row.lsf_sigma),
                             "sep_arcmin": float(row.get("sep_arcmin", np.nan)), "class": kind, "z": result.z,
                             "v_kms": C_KMS * result.z, "delta_chi2": result.delta_chi2_runner_up,
                             "chi2_reduced": result.chi2 / max(result.n_pixels - result.n_parameters, 1)})
    order = np.argsort([-r["snr"] for r in rows])
    panels = [panels[i] for i in order]
    index = pd.DataFrame([rows[i] for i in order])
    n_pages = int(np.ceil(len(panels) / per_page))
    ncols, nrows = 3, int(np.ceil(per_page / 3))
    with PdfPages(out_dir / "gallery.pdf") as pdf:
        for page in range(n_pages):
            chunk = panels[page * per_page: (page + 1) * per_page]
            fig, axes = plt.subplots(nrows, ncols, figsize=(14, 2.6 * nrows))
            for ax in np.atleast_1d(axes).ravel():
                ax.axis("off")
            for ax, item in zip(np.atleast_1d(axes).ravel(), chunk):
                ax.axis("on"); panel(ax, *item, host_v=host_v)
            fig.supxlabel("observed wavelength [Å]", fontsize=8)
            fig.supylabel(r"$F_\lambda$ [$10^{-17}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$]", fontsize=8)
            fig.suptitle(f"data (black, ±1σ grey) and our best fit (red) — page {page + 1}/{n_pages}, sorted by S/N", fontsize=9)
            fig.tight_layout(rect=(0.01, 0.01, 1, 0.96))
            pdf.savefig(fig)
            if page < 6:
                fig.savefig(out_dir / f"page{page + 1:02d}.png", dpi=dpi)
            plt.close(fig)
            index.loc[index.index[page * per_page: (page + 1) * per_page], "page"] = page + 1
    index.to_parquet(out_dir / "index.parquet", index=False)
    log.info("%d panels on %d pages in %.0f s -> %s", len(panels), n_pages, time.time() - started, out_dir)
    return index


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--min-snr", type=float, default=10.0)
    parser.add_argument("--max-objects", type=int, default=700)
    parser.add_argument("--per-page", type=int, default=12)
    parser.add_argument("--host-v", type=float, default=np.nan)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    c = pd.read_parquet(args.candidates)
    c = c[c.snr > args.min_snr].nlargest(args.max_objects, "snr")
    print(f"{len(c)} objects with S/N > {args.min_snr}: {(c.lsf_sigma < 16).sum()} unresolved, {(c.lsf_sigma >= 20).sum()} resolved")
    index = run(c, args.out_dir, per_page=args.per_page, host_v=args.host_v)
    print(index["class"].value_counts().to_dict())
    bad = index.chi2_reduced > 20
    print(f"median reduced chi2 {index.chi2_reduced.median():.2f}; unfittable (chi2r > 20): {bad.sum()} ({bad.mean():.0%})")
    res = index.lsf_sigma >= 20
    print(f"  unresolved: {(~res).sum()}, unfittable {int((bad & ~res).sum())} | resolved: {res.sum()}, unfittable {int((bad & res).sum())}")
    print(f"pages and gallery.pdf in {args.out_dir}")


if __name__ == "__main__":
    main()
