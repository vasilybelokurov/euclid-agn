"""Diagnostic atlas: spectra and fits as PNG figures.

Two audiences.  A *per-object* figure has to show enough for someone to decide
whether a candidate is real - data, both models, the broad component on its own,
and the residuals, because a Delta chi-squared built from a slow bend across the
whole range is continuum mismatch however large it is.  A *summary* figure has
to show where the statistic sits relative to the population, since no threshold
in this project comes from a chi-squared table.

Everything is written as PNG into a directory the caller chooses, by default
``plots/``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from euclid_agn.fit.screen import ScreenSettings, refine  # noqa: E402
from euclid_agn.io.sir import open_sir_file  # noqa: E402
from euclid_agn.numerics import blas_safe  # noqa: E402
from euclid_agn.plotting.diagnostics import (  # noqa: E402
    broad_on_continuum,
    plot_hypothesis_fit,
    plot_observation,
)

log = logging.getLogger(__name__)


def _window(spectrum, settings: ScreenSettings) -> np.ndarray:
    return (
        spectrum.usable()
        & (spectrum.wavelength >= settings.wavelength_min)
        & (spectrum.wavelength <= settings.wavelength_max)
    )


def _find_file(object_id: int, search_paths) -> tuple[str, object] | None:
    for path in search_paths:
        with open_sir_file(str(path)) as sir:
            if object_id in sir.object_ids():
                return str(path), sir.group_for_object(object_id)
    return None


@blas_safe
def plot_candidate(
    row,
    search_paths,
    directory: Path,
    settings: ScreenSettings = ScreenSettings(),
) -> Path | None:
    """Per-candidate figure: spectrum, M0, M1, broad component, residuals."""
    object_id = int(row["object_id"])
    found = _find_file(object_id, search_paths)
    if found is None:
        log.warning("object %s not found in the supplied files", object_id)
        return None
    path, _ = found
    with open_sir_file(path) as sir:
        spectrum = sir.read_combined(sir.group_for_object(object_id))
    sigma = row.get("broad_sigma_kms", np.nan)
    fit = refine(
        spectrum,
        float(row["z"]),
        str(row["system"]),
        float(sigma) if np.isfinite(sigma) else settings.broad_sigma_kms[0],
        settings=settings,
    )
    if fit is None:
        return None
    keep = _window(spectrum, settings)
    title = (
        f"object {object_id}   z={row['z']:.4f} ({row['system']})   "
        f"$\\Delta\\chi^2$={row['delta_chi2']:.0f}, effective {row['delta_chi2_refined_effective']:.0f}   "
        f"$\\chi^2_\\nu$(M0)={row.get('chi2_reduced_m0', float('nan')):.1f}   "
        f"orth={row.get('broad_continuum_orthogonality', float('nan')):.2f}"
    )
    return plot_hypothesis_fit(
        spectrum.wavelength[keep],
        spectrum.flux[keep],
        spectrum.variance[keep],
        fit,
        directory / f"fit_{object_id}.png",
        title=title,
    )


def plot_spectrum_page(object_id: int, search_paths, directory: Path) -> Path | None:
    """Per-object figure: combined spectrum above its contributing dithers."""
    found = _find_file(object_id, search_paths)
    if found is None:
        return None
    path, _ = found
    with open_sir_file(path) as sir:
        observation = sir.read_observation(object_id)
    return plot_observation(observation, directory / f"spectrum_{object_id}.png")


@blas_safe
def plot_fit_grid(
    rows: pd.DataFrame,
    search_paths,
    path: Path,
    settings: ScreenSettings = ScreenSettings(),
    n_columns: int = 2,
) -> Path:
    """One page of candidate fits, zoomed on the broad line under test.

    The zoom is deliberate: the question a reviewer asks of a candidate is
    whether the excess sits on the line, and a full-range panel hides that.
    """
    rows = rows.reset_index(drop=True)
    n = len(rows)
    n_rows = int(np.ceil(n / n_columns))
    fig, axes = plt.subplots(
        n_rows, n_columns, figsize=(5.6 * n_columns, 2.6 * n_rows), squeeze=False
    )
    for index, ax in enumerate(axes.ravel()):
        if index >= n:
            ax.axis("off")
            continue
        row = rows.iloc[index]
        object_id = int(row["object_id"])
        found = _find_file(object_id, search_paths)
        if found is None:
            ax.axis("off")
            continue
        path_found, _ = found
        with open_sir_file(path_found) as sir:
            spectrum = sir.read_combined(sir.group_for_object(object_id))
        sigma = row.get("broad_sigma_kms", np.nan)
        fit = refine(
            spectrum,
            float(row["z"]),
            str(row["system"]),
            float(sigma) if np.isfinite(sigma) else settings.broad_sigma_kms[0],
            settings=settings,
        )
        if fit is None:
            ax.axis("off")
            continue
        keep = _window(spectrum, settings)
        wavelength = spectrum.wavelength[keep]
        flux = spectrum.flux[keep]
        error = np.sqrt(spectrum.variance[keep])

        ax.fill_between(wavelength, flux - error, flux + error, color="0.87", lw=0)
        ax.plot(wavelength, flux, lw=0.7, color="0.35")
        ax.plot(wavelength, fit.m0.model, lw=1.1, color="tab:blue")
        if fit.m1 is not None:
            ax.plot(wavelength, fit.m1.model, lw=1.1, color="tab:red")
            ax.plot(
                wavelength, broad_on_continuum(fit), lw=0.9, ls="--", color="tab:red"
            )
        centre = _broad_centre(fit)
        if centre is not None:
            half = 8.0 * fit.broad.components()[0].observed_sigma_angstrom
            ax.set_xlim(max(wavelength[0], centre - half), min(wavelength[-1], centre + half))
            inside = (wavelength > centre - half) & (wavelength < centre + half)
            if np.any(inside):
                lo, hi = np.nanmin(flux[inside]), np.nanmax(flux[inside])
                pad = 0.2 * max(hi - lo, 1e-30)
                ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(
            f"{object_id}  z={row['z']:.3f}  "
            f"$\\Delta\\chi^2_{{\\rm eff}}$={row['delta_chi2_refined_effective']:.0f}  "
            f"FWHM={row.get('broad_fwhm_kms', float('nan')):.0f} km/s",
            fontsize=8,
        )
        ax.tick_params(labelsize=7)
    fig.supxlabel(r"observed wavelength [$\AA$]", fontsize=9)
    fig.supylabel(r"$f_\lambda$ [erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]", fontsize=9)
    fig.suptitle(
        "Stage-1 candidates: grey data with its $\\pm1\\sigma$ band, blue M0 "
        "(continuum + narrow), red M1 (all components), dashed red M1 continuum + broad",
        fontsize=10,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _broad_centre(fit) -> float | None:
    if fit.broad is None:
        return None
    components = fit.broad.visible_components(np.array([0.0, 1e9]))
    components = fit.broad.components()
    return components[0].centre if components else None


def plot_screen_summary(table: pd.DataFrame, path: Path) -> Path:
    """Where the statistic sits in the population, and what drives it.

    The point of this figure is that the detection statistic is not calibrated:
    the distribution shown is over essentially unselected spectra, so it is a
    first sketch of the empirical null, not a background to subtract.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5))

    statistic = np.asarray(table["delta_chi2_refined_effective"], dtype=float)
    ok = (
        np.asarray(table["continuum_model_ok"], dtype=bool)
        if "continuum_model_ok" in table
        else np.ones(statistic.size, dtype=bool)
    )

    ax = axes[0, 0]
    bins = np.logspace(-1, np.log10(max(statistic.max(), 10.0)), 40)
    ax.hist(np.clip(statistic, 0.1, None), bins=bins, color="0.7", label="all spectra")
    ax.hist(
        np.clip(statistic[ok], 0.1, None),
        bins=bins,
        color="tab:blue",
        label=r"$\chi^2_\nu$(M0) < 4",
    )
    ax.set_xscale("log")
    ax.axvline(25.0, color="crimson", ls="--", lw=1.2)
    ax.set_xlabel(r"effective $\Delta\chi^2$  ($\Delta\chi^2/\eta^2$)")
    ax.set_ylabel("spectra")
    ax.legend(fontsize=8)
    ax.text(
        0.03,
        0.95,
        "dashed line: nominal $\\Delta\\chi^2=25$\nsits near the 95th percentile\nof unselected spectra",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
    )

    ax = axes[0, 1]
    if "chi2_reduced_m0" in table:
        values = np.asarray(table["chi2_reduced_m0"], dtype=float)
        ax.hist(np.clip(values, 0, 12), bins=40, color="tab:blue")
        ax.axvline(np.nanmedian(values), color="crimson", ls="--", lw=1.2)
        ax.axvline(1.98, color="tab:green", ls=":", lw=1.4)
        ax.set_xlabel(r"reduced $\chi^2$ of M0")
        ax.set_ylabel("spectra")
        ax.text(
            0.97,
            0.95,
            f"median {np.nanmedian(values):.2f}\n"
            "green: independently measured\nnoise inflation $\\eta^2=1.98$",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
        )

    ax = axes[1, 0]
    if "broad_continuum_orthogonality" in table:
        values = np.asarray(table["broad_continuum_orthogonality"], dtype=float)
        finite = np.isfinite(values)
        ax.scatter(values[finite], statistic[finite], s=12, color="tab:blue", alpha=0.7)
        ax.set_yscale("log")
        ax.set_xlabel("broad-line orthogonality to the continuum")
        ax.set_ylabel(r"effective $\Delta\chi^2$")
        ax.text(
            0.03,
            0.95,
            "values near the 0.5 cut carry\nlittle independent width information",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
        )

    ax = axes[1, 1]
    if "broad_fwhm_kms" in table:
        values = np.asarray(table["broad_fwhm_kms"], dtype=float)
        finite = np.isfinite(values)
        ax.hist(values[finite], bins=30, color="tab:blue")
        ax.set_xlabel("recovered broad FWHM [km s$^{-1}$]")
        ax.set_ylabel("spectra")
        ax.text(
            0.97,
            0.95,
            "piling up at the widest\nallowed width is the signature\nof continuum degeneracy",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
        )

    fig.suptitle(
        f"Stage-1 statistics over {len(table)} spectra - an uncalibrated statistic, not a detection",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def build_atlas(
    table: pd.DataFrame,
    search_paths,
    directory: str | Path = "plots",
    n_candidates: int = 8,
    settings: ScreenSettings = ScreenSettings(),
    with_dithers: bool = True,
) -> dict[str, list[str]]:
    """Write the whole atlas and return the paths it created."""
    directory = Path(directory)
    (directory / "fits").mkdir(parents=True, exist_ok=True)
    (directory / "spectra").mkdir(parents=True, exist_ok=True)
    search_paths = [str(p) for p in search_paths]

    ranked = table.sort_values("delta_chi2_refined_effective", ascending=False)
    if "continuum_model_ok" in ranked:
        ranked = pd.concat(
            [ranked[ranked["continuum_model_ok"]], ranked[~ranked["continuum_model_ok"]]]
        )
    top = ranked.head(n_candidates)

    created: dict[str, list[str]] = {"fits": [], "spectra": [], "summary": []}
    for _, row in top.iterrows():
        made = plot_candidate(row, search_paths, directory / "fits", settings)
        if made:
            created["fits"].append(str(made))
        if with_dithers:
            page = plot_spectrum_page(int(row["object_id"]), search_paths, directory / "spectra")
            if page:
                created["spectra"].append(str(page))

    created["summary"].append(
        str(plot_fit_grid(top, search_paths, directory / "candidate_grid.png", settings))
    )
    created["summary"].append(
        str(plot_screen_summary(table, directory / "screen_statistics.png"))
    )
    return created


def plot_redshift_agreement(
    compared: pd.DataFrame,
    path: str | Path,
    reference_column: str = "spe_gal_z",
    reference_label: str = "SPE galaxy redshift",
    strength_column: str = "spe_best_snr",
    strength_label: str = "SPE best-line S/N",
    title: str | None = None,
) -> Path:
    """Blind-scan redshift against the SPE redshift, and where it fails.

    The comparison is between two measurements of the same photons, so
    agreement means the lines are real and the fit is useful - not that the
    object is an AGN.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    z_blind = np.asarray(compared["z"], dtype=float)
    z_reference = np.asarray(compared[reference_column], dtype=float)
    delta_v = np.asarray(compared["delta_v_kms"], dtype=float)
    snr = (
        np.asarray(compared[strength_column], dtype=float)
        if strength_column in compared
        else np.full(z_blind.size, np.nan)
    )
    agrees = np.asarray(compared["agrees"], dtype=bool)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

    ax = axes[0]
    ax.plot([0, 6], [0, 6], color="0.6", lw=0.9, ls="--")
    ax.scatter(z_reference[~agrees], z_blind[~agrees], s=14, color="0.7", label="disagrees")
    ax.scatter(z_reference[agrees], z_blind[agrees], s=16, color="tab:blue", label="agrees")
    ax.set_xlabel(reference_label)
    ax.set_ylabel("pipeline redshift")
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 6)
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[1]
    finite = np.isfinite(snr)
    ax.axhspan(-1000, 1000, color="0.9", lw=0)
    ax.scatter(snr[finite], np.clip(delta_v[finite], -2e5, 2e5), s=16, color="tab:blue")
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1000)
    ax.set_xlabel(strength_label)
    ax.set_ylabel(r"$\Delta v$ (pipeline $-$ reference) [km s$^{-1}$]")
    ax.text(
        0.03,
        0.05,
        "grey band: agreement within 1000 km/s",
        transform=ax.transAxes,
        fontsize=8,
    )

    ax = axes[2]
    if "snr_bin" in compared:
        grouped = compared.groupby("snr_bin", observed=True)["agrees"]
        fractions = grouped.mean()
        counts = grouped.size()
        positions = np.arange(len(fractions))
        ax.bar(positions, fractions.values, color="tab:blue")
        ax.set_xticks(positions)
        ax.set_xticklabels([str(i) for i in fractions.index], fontsize=8, rotation=20)
        for x, (fraction, count) in enumerate(zip(fractions.values, counts.values, strict=True)):
            ax.text(x, fraction + 0.02, f"n={count}", ha="center", fontsize=8)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("fraction agreeing with reference")
        ax.set_xlabel(strength_label)

    fig.suptitle(
        title or f"Redshift recovery on {len(compared)} real Q1 spectra",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path
