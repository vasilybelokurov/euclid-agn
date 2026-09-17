"""RV precision curves versus G and V from the dither-repeatability table.

V is derived from Gaia G and BP-RP with the Gaia (E)DR3 photometric
relationship G - V = f(BP-RP) (coefficients in ``G_MINUS_V``; source and
validity recorded there).  BP-RP is fetched from the Gaia archive if the table
lacks it.

Usage::

    python -m euclid_agn.validation.star_velocity_plot --table outputs/star_velocity.parquet
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

#: G - V = c0 + c1 x + c2 x^2 + c3 x^3, x = BP - RP.  Filled from the Gaia DR3 documentation
#: (Photometric relationships with other systems); see JOURNAL for the verified source.
G_MINUS_V: dict = {"coefficients": (-0.02704, 0.01424, -0.2156, 0.01426), "x_range": (-0.5, 5.0), "sigma": 0.03017,
                   "source": "Gaia DR3 documentation, Table 5.9 (Johnson-Cousins relationships; Riello et al. 2021): "
                             "https://gea.esac.esa.int/archive/documentation/GDR3/Data_processing/chap_cu5pho/"
                             "cu5pho_sec_photSystem/cu5pho_ssec_photRelations.html - VERIFIED 2026-09-17"}


def g_minus_v(bp_rp: np.ndarray) -> np.ndarray:
    c = G_MINUS_V["coefficients"]
    x = np.asarray(bp_rp, float)
    out = c[0] + c[1] * x + c[2] * x**2 + c[3] * x**3
    lo, hi = G_MINUS_V["x_range"]
    return np.where((x >= lo) & (x <= hi), out, np.nan)


def fetch_bp_rp(gaia_ids) -> pd.DataFrame:
    from euclid_agn.archive.tap import TapService, in_list_clause
    from euclid_agn.validation.star_velocity_experiment import GAIA_TAP

    tap = TapService(GAIA_TAP)
    ids = [int(x) for x in gaia_ids if np.isfinite(x) and x > 0]
    parts = []
    for i in range(0, len(ids), 400):
        parts.append(tap.query(f"SELECT source_id, phot_g_mean_mag, bp_rp, parallax, pm FROM gaiadr3.gaia_source WHERE {in_list_clause('source_id', ids[i:i + 400])}"))
    g = pd.concat(parts, ignore_index=True).rename(columns={"source_id": "gaia_id", "phot_g_mean_mag": "G_gaia"})
    g["gaia_id"] = g.gaia_id.astype(np.int64)
    return g


DITHER_ZERO_POINT_FLOOR_KMS = 110.0 / 2.0  # per-dither wavelength zero-point scatter 110 km/s rms, averaged over 4 dithers


def curves(t: pd.DataFrame, mag: str, edges) -> pd.DataFrame:
    """Per-magnitude-bin precision estimators from the combined-spectrum fits.

    ``formal_error``: rvspecfit's error on the combined spectrum (median);
    ``sigma_from_v``: 1.4826 x median |v| - an *upper* bound on the measurement
    error because it includes the stars' true velocity dispersion (a few tens
    of km/s for disk dwarfs at these latitudes); ``floor``: the wavelength
    zero-point systematic measured from dither cross-correlation.  The
    per-dither fits themselves are unusable (their wavelength solutions
    disagree by 0.4 pixel) and are not used.
    """
    rows = []
    ok = t[t[mag].notna() & t.comb_vel.notna()]
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = ok[(ok[mag] >= lo) & (ok[mag] < hi)]
        if len(s) < 8:
            continue
        rows.append({"mag": mag, "lo": lo, "hi": hi, "centre": 0.5 * (lo + hi), "n": len(s),
                     "formal_error": s.comb_vel_err.median(),
                     "formal_error_lo": s.comb_vel_err.quantile(0.25), "formal_error_hi": s.comb_vel_err.quantile(0.75),
                     "sigma_from_v": 1.4826 * s.comb_vel.abs().median(),
                     "frac_within_300": (s.comb_vel.abs() < 300).mean(),
                     "floor": DITHER_ZERO_POINT_FLOOR_KMS})
    return pd.DataFrame(rows)


def plot(cv: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(cv), figsize=(5.4 * len(cv), 4.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (mag, c) in zip(axes, cv.items()):
        ax.fill_between(c.centre, c.formal_error_lo, c.formal_error_hi, color="crimson", alpha=0.15)
        ax.plot(c.centre, c.formal_error, "o-", color="crimson", label="rvspecfit formal error, combined spectrum (median, IQR)")
        ax.plot(c.centre, c.sigma_from_v, "s--", color="tab:blue", label="1.48 × median |v|  (upper bound: includes true σ_v of the stars)")
        ax.axhline(c.floor.iloc[0], color="0.4", ls=":", label="wavelength zero-point floor (dither scatter 110 km/s / 2)")
        ax.axhline(259.0, color="0.7", ls="-.", lw=0.8, label="1 NISP pixel at 1.55 µm")
        for x, n in zip(c.centre, c.n):
            ax.text(x, 1.08 * c.sigma_from_v.max(), f"{n}", ha="center", fontsize=7)
        ax.set_yscale("log"); ax.set_ylim(20, 800); ax.set_xlabel(f"{mag.replace('_mag', '')} [mag]"); ax.grid(alpha=0.3, which="both")
        ax.set_title(f"Euclid NISP red grism: stellar RV precision vs {mag.replace('_mag', '')}", fontsize=9)
    axes[0].set_ylabel("σ(v) [km/s]"); axes[0].legend(fontsize=6.5, loc="upper left")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--table", type=Path, default=Path("outputs/star_velocity.parquet"))
    parser.add_argument("--plot", type=Path, default=Path("plots/star_rv_precision.png"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    t = pd.read_parquet(args.table)
    if "bp_rp" not in t.columns:
        g = fetch_bp_rp(t.gaia_id)
        t = t.merge(g[["gaia_id", "bp_rp"] + (["G_gaia"] if "G" not in t.columns else [])], on="gaia_id", how="left")
        if "G" not in t.columns:
            t = t.rename(columns={"G_gaia": "G"})
    t["V"] = t.G - g_minus_v(t.bp_rp)
    # stars without Gaia G (or outside the BP-RP validity range): G and V from Euclid I_E and H
    try:
        from euclid_agn.validation.star_photometry_calibration import COEF_PATH, euclid_g_v, mer_photometry

        if COEF_PATH.exists():
            need = t.G.isna() | t.V.isna()
            if need.any():
                mer = mer_photometry(t.loc[need, "object_id"])
                m = t.loc[need, ["object_id"]].merge(mer[["object_id", "I_E", "H_E"]], on="object_id", how="left")
                G_hat, V_hat = euclid_g_v(m.I_E.values, m.H_E.values)
                t.loc[need, "G"] = np.where(t.loc[need, "G"].isna(), G_hat, t.loc[need, "G"])
                t.loc[need, "V"] = np.where(t.loc[need, "V"].isna(), V_hat, t.loc[need, "V"])
                t.loc[need, "mag_source"] = "euclid_calibrated"
            t["mag_source"] = t.get("mag_source", pd.Series(index=t.index, dtype=object)).fillna("gaia")
    except Exception as exc:  # noqa: BLE001
        log.warning("photometric G/V fallback unavailable: %s", exc)
    t.to_parquet(args.table.with_name(args.table.stem + "_mags.parquet"), index=False)
    cv = {m: curves(t, m, edges) for m, edges in (("G", np.arange(13, 22, 1.0)), ("V", np.arange(13, 23, 1.0)), ("H_mag", np.arange(12, 20, 1.0))) if m in t}
    cv = {m: c for m, c in cv.items() if len(c)}
    pd.set_option("display.width", 200)
    for m, c in cv.items():
        print(f"--- {m}"); print(c[["lo", "hi", "n", "formal_error", "sigma_from_v", "frac_within_300"]].round(2).to_string(index=False))
    plot(cv, args.plot); print("wrote", args.plot)
    print(f"BP-RP median {t.bp_rp.median():.2f}; G-V median {(t.G - t.V).median():.2f}; V-H median {(t.V - t.H_mag).median():.2f}")


if __name__ == "__main__":
    main()
