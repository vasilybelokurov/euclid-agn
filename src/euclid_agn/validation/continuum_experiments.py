"""Continuum-redshift experiments on the DESI-matched low-z sample.

The question these answer is the user's requirement in plain terms: can the
redshift of a z < 0.9 galaxy be measured from a Euclid red-grism spectrum when
no emission line is available?  The truth is DESI; the baseline is Euclid's own
SPE redshift (16 % on this sample).  Every variant of the template basis,
polynomial, sign constraint and prior is run on the same objects with the same
pixels, so the comparison isolates the modelling choice.

Usage (paste-ready)::

    source ~/Work/venvs/.venv/bin/activate; export PYTHONPATH=src
    python -m euclid_agn.validation.continuum_experiments --limit 100 --out outputs/continuum_variants.parquet

Spectra come from the local cache; objects whose tile file is not cached are
skipped and counted.
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from euclid_agn.constants import C_KMS
from euclid_agn.fit.screen import ScreenSettings, prepare
from euclid_agn.fit.template_cube import CubeStore, cube_scan, redshift_grid
from euclid_agn.spectra.coherence import apply_coherence_mask
from euclid_agn.io.sir import open_sir_file
from euclid_agn.models.library import (
    DEFAULT_ROOT,
    Template,
    build_pca_basis,
    load_xsl_ssp_library,
)

log = logging.getLogger(__name__)

DEFAULT_CACHE = Path("~/data/euclid/q1/SIR").expanduser()
DEFAULT_SAMPLE = Path("outputs/desi_lowz_baseline.parquet")


@dataclass(frozen=True)
class Variant:
    """One modelling choice to test."""

    name: str
    basis: str = "pca"  # pca | archetypes
    n_components: int = 5
    poly_degree: int = 1
    nonnegative: bool = False
    log_age_min: float = 8.5
    mh_min: float = -0.5
    archetype_step: int = 1  # for basis=archetypes: take every k-th SSP of the (age, [M/H]) selection
    step_kms: float = 300.0
    z_max: float = 1.0
    nuisance: str = "poly"  # poly | spline (spline: n_knots B-spline projected out of data and templates)
    n_knots: int = 12
    coherence_mask: bool = False  # reject pixels the dithers disagree about before fitting
    coherence_threshold: float = 5.0
    systematic_fraction: float = 0.0  # fractional flux error added in quadrature (template/calibration floor)
    extras: dict = field(default_factory=dict)

    def templates(self, root=DEFAULT_ROOT) -> list[Template]:
        lib = load_xsl_ssp_library(root, log_age_min=self.log_age_min, mh_min=self.mh_min)
        if self.basis == "pca":
            return pca_templates(build_pca_basis(lib, n_components=self.n_components, wmax=19500.0))
        if self.basis == "archetypes":
            return lib[:: self.archetype_step]
        raise ValueError(self.basis)


def pca_templates(basis) -> list[Template]:
    """Mean + components of a PCA basis as Template objects (all on the basis grid)."""
    out = [Template("mean", basis.kind, basis.wavelength, basis.mean, np.ones(basis.wavelength.size, bool))]
    for i, c in enumerate(basis.components):
        out.append(Template(f"pc{i + 1}", basis.kind, basis.wavelength, c, np.ones(basis.wavelength.size, bool)))
    return out


def with_systematic_floor(spectrum, fraction: float):
    """Variance += (fraction * smoothed flux)^2: a per-pixel floor for template and calibration error.

    At S/N ~ 100 per pixel the archive variance makes a 3 % template mismatch
    a 3-sigma residual on every pixel; the floor keeps such broad-band
    mismatch from dominating the chi-squared over genuine spectral features.
    The smoothing (31-pixel median) stops the floor tracking noise spikes.
    """
    from dataclasses import replace
    from scipy.ndimage import median_filter

    ok = spectrum.usable()
    level = np.abs(median_filter(np.where(ok, spectrum.flux, np.nanmedian(spectrum.flux[ok])), size=31, mode="nearest"))
    metadata = {**spectrum.metadata, "systematic_fraction": float(fraction)}
    return replace(spectrum, variance=spectrum.variance + (fraction * level) ** 2, metadata=metadata)


def load_sample(path: Path = DEFAULT_SAMPLE) -> pd.DataFrame:
    t = pd.read_parquet(path)
    t = t.drop_duplicates("object_id")
    cols = ["object_id", "tile_id", "desi_z", "desi_spectype", "spe_gal_z", "spe_class", "phz_median",
            "median_snr_per_pixel", "lsf_sigma"]
    return t[[c for c in cols if c in t.columns]].reset_index(drop=True)


def object_index(cache: Path = DEFAULT_CACHE, index_path: Path | None = None) -> pd.DataFrame:
    """object_id -> cached file, from a header scan of every cached SIR file.

    The association table's tile id does not identify the file an object was
    extracted into (tiles can have several files and objects sit in overlaps),
    so the index is built from the files themselves and cached as parquet.
    """
    index_path = index_path or Path("outputs/cache_object_index.parquet")
    files = sorted(cache.glob("*/EUC_SIR_W-COMBSPEC_*.fits"))
    if index_path.exists():
        idx = pd.read_parquet(index_path)
        if set(idx["file"].unique()) == {str(f) for f in files}:
            return idx
    rows = []
    for f in files:
        with open_sir_file(f) as handle:
            rows.extend({"object_id": oid, "file": str(f), "tile_id": handle.tile_id} for oid in handle.object_ids())
    idx = pd.DataFrame(rows).drop_duplicates("object_id")
    index_path.parent.mkdir(parents=True, exist_ok=True)
    idx.to_parquet(index_path, index=False)
    return idx


def iter_spectra(sample: pd.DataFrame, cache: Path = DEFAULT_CACHE, with_dithers: bool = False):
    """Yield (row, combined Spectrum1D [, observation]), grouped by file so each opens once."""
    idx = object_index(cache).set_index("object_id")["file"]
    located = sample.assign(file=sample["object_id"].map(idx))
    missing = located["file"].isna().sum()
    if missing:
        log.warning("%d of %d objects are not in any cached file", missing, len(sample))
    for path, rows in located.dropna(subset=["file"]).groupby("file"):
        with open_sir_file(path) as f:
            for _, row in rows.iterrows():
                obs = f.read_observation(int(row["object_id"]), with_dithers=with_dithers)
                yield (row, obs.combined, obs) if with_dithers else (row, obs.combined)


def run_variants(
    sample: pd.DataFrame,
    variants: list[Variant],
    cache: Path = DEFAULT_CACHE,
    settings: ScreenSettings | None = None,
    tolerance_kms: float = 1000.0,
    root=DEFAULT_ROOT,
) -> pd.DataFrame:
    """Per-object rows for every variant, with agreement against DESI."""
    settings = settings or ScreenSettings(n_knots=1, outlier_threshold=5.0)
    stores: dict[str, CubeStore] = {}
    rows = []
    started = time.time()
    n_spectra = 0
    need_dithers = any(v.coherence_mask for v in variants)
    for item in iter_spectra(sample, cache, with_dithers=need_dithers):
        row, spectrum = item[0], item[1]
        observation = item[2] if need_dithers else None
        prepared = {}  # by (n_knots, masked): spline-nuisance variants need their own continuum basis
        masked_spectrum, n_bad = None, 0
        if need_dithers:
            masked_spectrum, report = apply_coherence_mask(observation, threshold=max(v.coherence_threshold for v in variants))
            n_bad = report.n_bad if report is not None else 0
        n_spectra += 1
        for v in variants:
            if v.name not in stores:
                stores[v.name] = CubeStore({"GALAXY": v.templates(root)}, redshift_grid(0.0, v.z_max, v.step_kms),
                                           spectrum.wavelength, spectrum.bin_width)
            cube = stores[v.name].get("GALAXY", spectrum.lsf_sigma)
            source = masked_spectrum if v.coherence_mask else spectrum
            if v.systematic_fraction > 0:
                source = with_systematic_floor(source, v.systematic_fraction)
            key = (v.n_knots if v.nuisance == "spline" else 1, v.coherence_mask, v.systematic_fraction)
            if key not in prepared:
                prepared[key] = prepare(source, ScreenSettings(**{**settings.__dict__, "n_knots": key[0]}))
            this = prepared[key]
            if this is None:
                continue
            res = cube_scan(source, this, cube, poly_degree=v.poly_degree, nonnegative=v.nonnegative,
                            z_prior=float(row.get("phz_median", np.nan)), spline_nuisance=v.nuisance == "spline")
            if res is None:
                continue
            desi_z = float(row["desi_z"])
            out = {"variant": v.name, "object_id": int(row["object_id"]), "desi_z": desi_z,
                   "snr": float(row.get("median_snr_per_pixel", np.nan)), "lsf_sigma": float(spectrum.lsf_sigma),
                   "spe_z": float(row.get("spe_gal_z", np.nan)), "phz": float(row.get("phz_median", np.nan))}
            out.update(res.as_row())
            out["chi2_red"] = res.chi2 / max(res.n_pixels - res.n_parameters, 1)
            out["dv"] = C_KMS * (res.z - desi_z) / (1 + desi_z)
            out["agree"] = abs(out["dv"]) < tolerance_kms
            out["agree_001"] = abs(res.z - desi_z) / (1 + desi_z) < 0.01  # the catastrophic-failure convention
            out["n_coherence_masked"] = n_bad if v.coherence_mask else 0
            out["dv_prior"] = C_KMS * (res.z_prior - desi_z) / (1 + desi_z) if np.isfinite(res.z_prior) else np.nan
            out["agree_prior"] = abs(out["dv_prior"]) < tolerance_kms if np.isfinite(out["dv_prior"]) else False
            rows.append(out)
    log.info("%d spectra x %d variants in %.1f s; %d cubes built", n_spectra, len(variants), time.time() - started,
             sum(len(s) for s in stores.values()))
    return pd.DataFrame(rows)


def summarise(table: pd.DataFrame, cuts=(0, 10, 25, 50, 100)) -> pd.DataFrame:
    """Agreement per variant overall, by redshift bin, and purity at Delta chi2 cuts."""
    out = []
    z_bins = [(0.0, 0.15), (0.15, 0.3), (0.3, 0.45), (0.45, 0.9)]
    for name, g in table.groupby("variant", sort=False):
        row = {"variant": name, "n": len(g), "agree": g["agree"].mean(), "agree_001": g["agree_001"].mean(),
               "agree_prior": g["agree_prior"].mean(), "chi2_red_median": g["chi2_red"].median()}
        for lo, hi in z_bins:
            s = g[(g.desi_z >= lo) & (g.desi_z < hi)]
            row[f"z{lo:.2f}-{hi:.2f}"] = s["agree"].mean() if len(s) else np.nan
        for c in cuts:
            s = g[g["cz_delta_chi2_runner_up"] > c]
            row[f"purity>{c}"] = s["agree"].mean() if len(s) else np.nan
            row[f"kept>{c}"] = len(s) / len(g)
        out.append(row)
    return pd.DataFrame(out)


DEFAULT_VARIANTS = [
    Variant("pca5_p1"),
    Variant("pca3_p1", n_components=3),
    Variant("pca2_p1", n_components=2),
    Variant("pca5_p0", poly_degree=0),
    Variant("pca5_p2", poly_degree=2),
    Variant("pca8_p1", n_components=8),
    Variant("arch_nnls_p1", basis="archetypes", nonnegative=True, archetype_step=6),
    Variant("arch_nnls_p1_young", basis="archetypes", nonnegative=True, archetype_step=6, log_age_min=7.7),
    Variant("arch_nnls_p3", basis="archetypes", nonnegative=True, archetype_step=6, poly_degree=3),
    Variant("arch_nnls_spline12", basis="archetypes", nonnegative=True, archetype_step=6, nuisance="spline", n_knots=12),
    Variant("arch_nnls_spline6", basis="archetypes", nonnegative=True, archetype_step=6, nuisance="spline", n_knots=6),
    Variant("pca5_spline12", nuisance="spline", n_knots=12),
    Variant("arch_all_nnls_p1", basis="archetypes", nonnegative=True, archetype_step=1),
    Variant("arch_nnls_p1_coh", basis="archetypes", nonnegative=True, archetype_step=6, coherence_mask=True),
    Variant("arch_nnls_p1_coh3", basis="archetypes", nonnegative=True, archetype_step=6, coherence_mask=True, coherence_threshold=3.0),
    Variant("arch_nnls_p3_coh", basis="archetypes", nonnegative=True, archetype_step=6, poly_degree=3, coherence_mask=True),
    Variant("arch_nnls_p1_sys02", basis="archetypes", nonnegative=True, archetype_step=6, systematic_fraction=0.02),
    Variant("arch_nnls_p1_sys05", basis="archetypes", nonnegative=True, archetype_step=6, systematic_fraction=0.05),
    Variant("arch_nnls_p1_coh_sys03", basis="archetypes", nonnegative=True, archetype_step=6, coherence_mask=True, systematic_fraction=0.03),
    Variant("arch_nnls_spline12_coh", basis="archetypes", nonnegative=True, archetype_step=6, nuisance="spline", n_knots=12, coherence_mask=True),
    Variant("pca5_p2_coh_sys03", poly_degree=2, coherence_mask=True, systematic_fraction=0.03),
]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--variants", nargs="*", default=None, help="names from DEFAULT_VARIANTS")
    parser.add_argument("--out", type=Path, default=Path("outputs/continuum_variants.parquet"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sample = load_sample(args.sample)
    if args.limit:
        sample = sample.iloc[: args.limit]
    variants = DEFAULT_VARIANTS if not args.variants else [v for v in DEFAULT_VARIANTS if v.name in args.variants]
    table = run_variants(sample, variants, cache=args.cache)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.out, index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print(summarise(table).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
