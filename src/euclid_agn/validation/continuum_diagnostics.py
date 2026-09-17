"""Why does a continuum redshift fail?  Per-object diagnostics against DESI.

For every object the scan's chi-squared curve is compared with its value at
the DESI redshift: ``dchi2_true = chi2(z_DESI) - chi2_min``.  If failures have
dchi2_true of a few, the data cannot tell the redshifts apart (information
limit); if hundreds, the model is systematically wrong somewhere (calibration,
extraction, templates) and the winner is an artefact.  The worst high-S/N
failures are plotted: data, best model, model forced to the DESI redshift and
the chi-squared curve.

Usage::

    source ~/Work/venvs/.venv/bin/activate; export PYTHONPATH=src
    python -m euclid_agn.validation.continuum_diagnostics --variant arch_nnls_p1 --plot-worst 6
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from euclid_agn.constants import C_KMS
from euclid_agn.fit.screen import ScreenSettings, prepare
from euclid_agn.fit.template_cube import CubeStore, cube_scan, fit_at, model_flux, redshift_grid
from euclid_agn.validation.continuum_experiments import DEFAULT_CACHE, DEFAULT_VARIANTS, Variant, iter_spectra, load_sample

log = logging.getLogger(__name__)


def scan_with_truth(spectrum, projected, cube, variant: Variant, desi_z: float, window_kms: float = 600.0):
    """Full scan plus the chi-squared minimum within ``window_kms`` of the DESI redshift."""
    kw = {"poly_degree": variant.poly_degree, "nonnegative": variant.nonnegative,
          "spline_nuisance": variant.nuisance == "spline"}
    res = cube_scan(spectrum, projected, cube, **kw)
    if res is None:
        return None, None
    near = np.abs(C_KMS * (res.grid_z - desi_z) / (1 + desi_z)) < window_kms
    chi2_true = float(np.nanmin(res.grid_chi2[near])) if np.isfinite(res.grid_chi2[near]).any() else np.nan
    return res, chi2_true


def run(sample: pd.DataFrame, variant: Variant, cache=DEFAULT_CACHE, tolerance_kms: float = 1000.0,
        plot_worst: int = 0, plot_dir: Path = Path("plots/continuum_failures")) -> pd.DataFrame:
    settings = ScreenSettings(n_knots=variant.n_knots if variant.nuisance == "spline" else 1, outlier_threshold=5.0)
    store = CubeStore({"GALAXY": variant.templates()}, redshift_grid(0.0, variant.z_max, variant.step_kms),
                      None, 13.4)
    rows, cache_for_plots = [], {}
    for row, spectrum in iter_spectra(sample, cache):
        if store.wavelength is None or store.wavelength.size == 0:
            store.wavelength = spectrum.wavelength
        projected = prepare(spectrum, settings)
        if projected is None:
            continue
        cube = store.get("GALAXY", spectrum.lsf_sigma)
        desi_z = float(row["desi_z"])
        res, chi2_true = scan_with_truth(spectrum, projected, cube, variant, desi_z)
        if res is None:
            continue
        dv = C_KMS * (res.z - desi_z) / (1 + desi_z)
        out = {"object_id": int(row["object_id"]), "desi_z": desi_z, "z_best": res.z, "dv": dv,
               "agree": abs(dv) < tolerance_kms, "chi2_min": res.chi2, "chi2_true": chi2_true,
               "dchi2_true": chi2_true - res.chi2, "dchi2_runner": res.delta_chi2_runner_up,
               "chi2_red": res.chi2 / max(res.n_pixels - res.n_parameters, 1), "n_pixels": res.n_pixels,
               "snr": float(row.get("median_snr_per_pixel", np.nan)), "lsf_sigma": float(spectrum.lsf_sigma),
               "phz": float(row.get("phz_median", np.nan)), "spe_z": float(row.get("spe_gal_z", np.nan))}
        rows.append(out)
        if plot_worst:
            cache_for_plots[out["object_id"]] = (spectrum, projected, cube, res)
    table = pd.DataFrame(rows)
    if plot_worst and not table.empty:
        worst = table[~table.agree].sort_values("snr", ascending=False).head(plot_worst)
        plot_dir.mkdir(parents=True, exist_ok=True)
        for _, r in worst.iterrows():
            spectrum, projected, cube, res = cache_for_plots[r.object_id]
            plot_object(spectrum, projected, cube, res, variant, r, plot_dir / f"fail_{r.object_id}.png")
    return table


def plot_object(spectrum, projected, cube, res, variant: Variant, r, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kw = {"poly_degree": variant.poly_degree, "nonnegative": variant.nonnegative,
          "spline_nuisance": variant.nuisance == "spline"}
    at_truth = fit_at(spectrum, projected, cube, r.desi_z, **kw)
    keep = np.isin(spectrum.wavelength, projected.wavelength)
    w = projected.wavelength
    flux = spectrum.flux[keep]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 7.5), gridspec_kw={"height_ratios": [2, 1]})
    scale = 1e17
    ax.plot(w, flux * scale, color="k", lw=0.7, label="data (kept pixels)")
    ax.plot(w, model_flux(spectrum, projected, cube, res, variant.poly_degree, kw["spline_nuisance"]) * scale,
            color="crimson", lw=1.2, label=f"best fit z={res.z:.3f}  chi2={res.chi2:.0f}")
    if at_truth is not None:
        ax.plot(w, model_flux(spectrum, projected, cube, at_truth, variant.poly_degree, kw["spline_nuisance"]) * scale,
                color="tab:blue", lw=1.2, ls="--", label=f"forced to DESI z={r.desi_z:.3f}  chi2={at_truth.chi2:.0f}")
    ax.set_ylabel(r"$F_\lambda$ [$10^{-17}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$]")
    ax.set_title(f"object {r.object_id}: S/N {r.snr:.0f}/pix, LSF sigma {r.lsf_sigma:.0f} Å, PHZ {r.phz:.2f}, SPE {r.spe_z:.2f}; variant {variant.name}", fontsize=9)
    ax.legend(fontsize=8, loc="best")
    ax2.plot(res.grid_z, res.grid_chi2 - res.chi2, color="k", lw=0.9)
    ax2.axvline(r.desi_z, color="tab:blue", ls="--", label="DESI z")
    ax2.axvline(res.z, color="crimson", label="best")
    ax2.set_ylim(-5, min(np.nanmax(res.grid_chi2 - res.chi2), 20 * max(r.dchi2_true, 25)))
    ax2.set_xlabel("trial redshift"); ax2.set_ylabel(r"$\chi^2 - \chi^2_{\min}$"); ax2.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def summarise(table: pd.DataFrame) -> str:
    lines = []
    for label, sel in (("right", table.agree), ("wrong", ~table.agree)):
        s = table[sel]
        q = s["dchi2_true"].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).round(1).tolist()
        lines.append(f"{label:5s} n={len(s):3d}  dchi2_true quantiles 10/25/50/75/90%: {q}  median chi2_red {s.chi2_red.median():.2f}")
    wrong = table[~table.agree]
    for lo, hi in ((0, 10), (10, 30), (30, 100), (100, 1e9)):
        s = wrong[(wrong.dchi2_true >= lo) & (wrong.dchi2_true < hi)]
        lines.append(f"  wrong with {lo}<=dchi2_true<{hi}: {len(s)} ({len(s)/max(len(wrong),1):.0%}), median S/N {s.snr.median():.0f}")
    return "\n".join(lines)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--variant", default="arch_nnls_p1")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--plot-worst", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    variant = {v.name: v for v in DEFAULT_VARIANTS}[args.variant]
    sample = load_sample()
    if args.limit:
        sample = sample.iloc[: args.limit]
    table = run(sample, variant, plot_worst=args.plot_worst)
    out = args.out or Path(f"outputs/continuum_diagnostics_{variant.name}.parquet")
    table.to_parquet(out, index=False)
    print(summarise(table))


if __name__ == "__main__":
    main()
