"""G and V for Euclid point sources without a Gaia match, from Euclid photometry.

Training set: the Gaia-matched point sources of ``outputs/star_truth.parquet``
(MER I_E, Y, J, H aperture fluxes from IRSA; Gaia G and BP-RP from the ESA
archive; V from G via the Gaia DR3 relation).  Fits

    G = I_E + P_G(I_E - H),   V = I_E + P_V(I_E - H)

with low-order polynomials, records the residual scatter per colour bin and
writes the coefficients to ``outputs/euclid_to_gaia_v.json`` for
:func:`euclid_g_v` to use on any star.  I_E - H spans the K/M dwarf sequence
well; the colour term is what carries the temperature information.

Usage::

    python -m euclid_agn.validation.star_photometry_calibration
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from euclid_agn.archive import schema
from euclid_agn.archive.tap import TapService, in_list_clause, quote_columns
from euclid_agn.validation.star_velocity_plot import g_minus_v

log = logging.getLogger(__name__)
GAIA_TAP = "https://gea.esac.esa.int/tap-server/tap/sync"
COEF_PATH = Path("outputs/euclid_to_gaia_v.json")


def mer_photometry(object_ids, chunk: int = 400) -> pd.DataFrame:
    tap = TapService(schema.IRSA_TAP_SYNC)
    cols = ["object_id", "flux_vis_2fwhm_aper", "flux_y_2fwhm_aper", "flux_j_2fwhm_aper", "flux_h_2fwhm_aper"]
    ids = [int(x) for x in object_ids]
    frames = [tap.query(f"SELECT {quote_columns(cols)} FROM {schema.MER_CATALOGUE} WHERE {in_list_clause('object_id', ids[i:i + chunk])}") for i in range(0, len(ids), chunk)]
    t = pd.concat(frames, ignore_index=True)
    t["object_id"] = t.object_id.astype(np.int64)
    for band, col in (("I_E", "flux_vis_2fwhm_aper"), ("Y_E", "flux_y_2fwhm_aper"), ("J_E", "flux_j_2fwhm_aper"), ("H_E", "flux_h_2fwhm_aper")):
        t[band] = 23.9 - 2.5 * np.log10(t[col].where(t[col] > 0))
    return t


def gaia_photometry(gaia_ids, chunk: int = 400) -> pd.DataFrame:
    tap = TapService(GAIA_TAP)
    ids = [int(x) for x in gaia_ids if np.isfinite(x) and x > 0]
    frames = [tap.query(f"SELECT source_id, phot_g_mean_mag, bp_rp, parallax, pm, ruwe FROM gaiadr3.gaia_source WHERE {in_list_clause('source_id', ids[i:i + chunk])}") for i in range(0, len(ids), chunk)]
    g = pd.concat(frames, ignore_index=True).rename(columns={"source_id": "gaia_id", "phot_g_mean_mag": "G"})
    g["gaia_id"] = g.gaia_id.astype(np.int64)
    g["V"] = g.G - g_minus_v(g.bp_rp)
    return g


def fit_relation(colour: np.ndarray, target: np.ndarray, degree: int = 3, clip: float = 3.0):
    """Polynomial target = P(colour) with iterative sigma clipping; returns coefficients, sigma, n."""
    ok = np.isfinite(colour) & np.isfinite(target)
    c, y = colour[ok], target[ok]
    keep = np.ones(c.size, bool)
    for _ in range(5):
        p = np.polyfit(c[keep], y[keep], degree)
        resid = y - np.polyval(p, c)
        sigma = 1.4826 * np.median(np.abs(resid[keep] - np.median(resid[keep])))
        new = np.abs(resid) < clip * sigma
        if new.sum() == keep.sum():
            break
        keep = new
    full = np.full(colour.size, np.nan)
    full[ok] = resid  # residuals aligned with the input rows
    return p, float(sigma), int(keep.sum()), full


def euclid_g_v(I_E, H_E, coefficients: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """G and V from Euclid I_E and H using the stored calibration (NaN outside the colour range)."""
    coef = coefficients or json.loads(COEF_PATH.read_text())
    colour = np.asarray(I_E, float) - np.asarray(H_E, float)
    lo, hi = coef["colour_range"]
    inside = (colour >= lo) & (colour <= hi)
    G = np.asarray(I_E, float) + np.polyval(coef["G_minus_I"], colour)
    V = np.asarray(I_E, float) + np.polyval(coef["V_minus_I"], colour)
    return np.where(inside, G, np.nan), np.where(inside, V, np.nan)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--stars", type=Path, default=Path("outputs/star_truth.parquet"))
    parser.add_argument("--out", type=Path, default=COEF_PATH)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--from-cache", action="store_true", help="reuse outputs/star_photometry_training.parquet")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.from_cache:
        t = pd.read_parquet("outputs/star_photometry_training.parquet")
    else:
        stars = pd.read_parquet(args.stars)
        mer = mer_photometry(stars.object_id)
        gaia = gaia_photometry(stars.gaia_id)
        t = stars[["object_id", "gaia_id", "point_like_prob"]].merge(mer, on="object_id").merge(gaia, on="gaia_id")
        t = t[t.point_like_prob > 0.99]
        t["colour"] = t.I_E - t.H_E
        t.to_parquet("outputs/star_photometry_training.parquet", index=False)
    pG, sG, nG, rG = fit_relation(t.colour.values, (t.G - t.I_E).values, args.degree)
    pV, sV, nV, rV = fit_relation(t.colour.values, (t.V - t.I_E).values, args.degree)
    lo, hi = np.nanpercentile(t.colour, [1, 99])
    coef = {"G_minus_I": pG.tolist(), "V_minus_I": pV.tolist(), "colour": "I_E - H_E (MER 2-FWHM apertures)",
            "colour_range": [float(lo), float(hi)], "sigma_G": sG, "sigma_V": sV, "n_G": nG, "n_V": nV,
            "degree": args.degree, "training": "Gaia DR3-matched Euclid Q1 point sources, point_like_prob > 0.99",
            "V_source": "G - (G-V)(BP-RP), Gaia DR3 documentation Table 5.9"}
    args.out.write_text(json.dumps(coef, indent=2))
    print(f"training stars: {len(t)}; colour range (1-99%): {lo:.2f}-{hi:.2f}; G range {t.G.min():.1f}-{t.G.max():.1f}")
    print(f"G - I_E = P(I_E - H): sigma {sG:.3f} mag (n={nG}); V - I_E: sigma {sV:.3f} mag (n={nV})")
    for c0, c1 in ((0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.5)):
        s = t[(t.colour >= c0) & (t.colour < c1)]
        if len(s) > 10:
            sel = ((t.colour >= c0) & (t.colour < c1)).values
            print(f"  I_E-H {c0}-{c1}: n={len(s):4d}  median G-I_E {np.median(s.G - s.I_E):+.2f}  V-I_E {np.nanmedian(s.V - s.I_E):+.2f}  resid rms G {np.nanstd(rG[sel]):.3f}  V {np.nanstd(rV[sel]):.3f}")
    # the same stars, without using Gaia: apply the calibration and compare
    G_hat, V_hat = euclid_g_v(t.I_E.values, t.H_E.values, coef)
    print(f"round-trip: |G_hat - G| median {np.nanmedian(np.abs(G_hat - t.G)):.3f}, |V_hat - V| median {np.nanmedian(np.abs(V_hat - t.V)):.3f}")


if __name__ == "__main__":
    main()
