"""Diagnostic plots.

Conventions: observed wavelength in Angstrom on x, flux density in
erg s^-1 cm^-2 Angstrom^-1 on y, a common x range across panels so dithers can
be compared by eye, unusable pixels shown as a shaded band rather than silently
dropped, and the region outside the 12500-18500 Angstrom science window hatched.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from euclid_agn.constants import (  # noqa: E402
    RGS_SCIENCE_WMAX_ANGSTROM,
    RGS_SCIENCE_WMIN_ANGSTROM,
)


def broad_on_continuum(fit) -> np.ndarray:
    """M1 with its narrow lines removed: continuum plus the broad component.

    Plotting the broad component on top of *M0's* continuum would be wrong, and
    visibly so: M1 refits the continuum and the narrow lines in the presence of
    the broad component, so the two continua differ.
    """
    design = fit.blocks_m1.design
    coefficients = np.asarray(fit.m1.coefficients, dtype=float).copy()
    narrow = fit.blocks_m1.slices.get("narrow")
    if narrow is not None:
        coefficients[narrow] = 0.0
    return design @ coefficients


def _finite_limits(values: np.ndarray, pad: float = 0.1) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return (-1.0, 1.0)
    lo, hi = float(np.min(finite)), float(np.max(finite))
    if hi <= lo:
        hi = lo + max(abs(lo), 1.0) * 1e-3
    span = hi - lo
    return (lo - pad * span, hi + pad * span)


def plot_spectrum(ax, spectrum, label: str = "", show_sigma: bool = True) -> None:
    """One spectrum panel: usable flux, its 1-sigma band, and unusable pixels."""
    ok = spectrum.usable()
    w = np.asarray(spectrum.wavelength)
    f = np.where(ok, spectrum.flux, np.nan)
    sigma = np.where(ok, np.sqrt(spectrum.variance), np.nan)

    ylim = _finite_limits(np.concatenate([f - sigma, f + sigma]))
    ax.fill_between(
        w,
        ylim[0],
        ylim[1],
        where=~ok,
        color="0.88",
        lw=0,
        step="mid",
        zorder=0,
        label="unusable pixel (masked or bad variance)",
    )
    ax.axvspan(
        w[0] - 200.0,
        RGS_SCIENCE_WMIN_ANGSTROM,
        color="0.96",
        lw=0,
        zorder=-1,
        label=f"outside {RGS_SCIENCE_WMIN_ANGSTROM:.0f}-{RGS_SCIENCE_WMAX_ANGSTROM:.0f} $\\AA$",
    )
    ax.axvspan(RGS_SCIENCE_WMAX_ANGSTROM, w[-1] + 200.0, color="0.96", lw=0, zorder=-1)
    if show_sigma:
        ax.fill_between(
            w,
            f - sigma,
            f + sigma,
            alpha=0.3,
            lw=0,
            color="tab:blue",
            zorder=2,
            label=r"$\pm1\sigma$ (reported variance)",
        )
    ax.plot(w, f, lw=0.9, color="tab:blue", zorder=3)
    ax.set_ylim(*ylim)
    ax.set_xlim(w[0] - 200.0, w[-1] + 200.0)
    if label:
        ax.text(
            0.99,
            0.93,
            label,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
        )


def plot_observation(observation, path: str | Path, title: str | None = None) -> Path:
    """Combined spectrum on top, individual dithers below, shared x range."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 1 + observation.n_dithers
    fig, axes = plt.subplots(n, 1, figsize=(10, 1.9 * n), sharex=True)
    axes = np.atleast_1d(axes)

    combined = observation.combined
    metrics = combined.quality_metrics()
    plot_spectrum(
        axes[0],
        combined,
        label=(
            f"combined  N_dith$\\leq${observation.n_dithers}  "
            f"usable {metrics['usable_pixel_fraction']:.1%}  "
            f"LSF $\\sigma$={combined.lsf_sigma:.1f} $\\AA$"
        ),
    )
    for ax, dither in zip(axes[1:], observation.dithers, strict=False):
        plot_spectrum(
            ax,
            dither,
            label=(
                f"dither {dither.dither_id}  {dither.gwa_position}  "
                f"{dither.n_contaminants} contaminants  "
                f"usable {dither.quality_metrics()['usable_pixel_fraction']:.1%}"
            ),
        )
    for ax in axes:
        ax.set_ylabel(r"$f_\lambda$")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, fontsize=7, loc="upper left", framealpha=0.9)
    axes[-1].set_xlabel(r"observed wavelength [$\AA$]")
    fig.suptitle(
        title or f"object {observation.object_id}   tile {observation.source.tile_id}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_availability(table, path: str | Path, min_usable_fraction: float = 0.5) -> Path:
    """Four-panel view of what the Q1 parent sample actually looks like.

    Panels: usable-pixel fraction inside the science window, continuum S/N,
    effective LSF width, and number of contributing dithers.  These are the
    axes along which the selection function has to be reported, so this figure
    is the honest picture of what is available before any AGN search begins.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    usable = np.asarray(table["usable_fraction_science"], dtype=float)
    snr = np.asarray(table["median_snr_science"], dtype=float)
    lsf = np.asarray(table["lsf_sigma"], dtype=float)
    ndith = np.asarray(table["n_dither_hdus"], dtype=float)
    fittable = usable >= min_usable_fraction

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5))

    ax = axes[0, 0]
    ax.hist(usable, bins=np.linspace(0, 1, 41), color="tab:blue")
    ax.axvline(min_usable_fraction, color="crimson", lw=1.2, ls="--")
    ax.set_xlabel("usable pixel fraction (12500-18500 $\\AA$)")
    ax.set_ylabel("spectra")
    ax.text(
        0.5,
        0.92,
        f"{fittable.mean():.0%} above threshold\n{(usable <= 0).mean():.0%} fully masked",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
    )

    ax = axes[0, 1]
    finite = np.isfinite(snr) & fittable
    ax.hist(np.clip(snr[finite], -2, 60), bins=60, color="tab:blue")
    ax.axvline(3.0, color="crimson", lw=1.2, ls="--")
    ax.set_xlabel("median continuum S/N per pixel")
    ax.set_ylabel("spectra")
    ax.text(
        0.95,
        0.92,
        f"fittable & S/N>3: {(fittable & (snr > 3)).mean():.0%}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
    )

    ax = axes[1, 0]
    ax.hist(np.clip(lsf, 0, 80), bins=60, color="tab:blue")
    ax.axvline(np.nanmedian(lsf), color="crimson", lw=1.2, ls="--")
    ax.set_xlabel(r"effective LSF $\sigma$ [$\AA$]")
    ax.set_ylabel("spectra")
    ax.text(
        0.95,
        0.92,
        f"median {np.nanmedian(lsf):.1f} $\\AA$\n"
        f"$R\\approx${15000 / (2.355 * np.nanmedian(lsf)):.0f} at 1.5 $\\mu$m",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
    )

    ax = axes[1, 1]
    values, counts = np.unique(ndith[np.isfinite(ndith)], return_counts=True)
    ax.bar(values, counts, color="tab:blue")
    ax.set_xlabel("contributing dithers")
    ax.set_ylabel("spectra")
    ax.text(
        0.95,
        0.92,
        f"{(ndith >= 4).mean():.0%} have $\\geq$4 dithers",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
    )

    fig.suptitle(
        f"Euclid Q1 spectrum availability: {len(usable)} spectra, "
        f"{len(np.unique(table['tile_id']))} tiles",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_hypothesis_fit(
    wavelength,
    flux,
    variance,
    fit,
    path: str | Path,
    title: str | None = None,
) -> Path:
    """Data with the M0 and M1 models overlaid, and the normalised residuals.

    The point of the figure is to show *where* the broad component is doing
    work. A Δχ² that comes from a handful of pixels far from the line, or from
    a slow bend across the whole range, is continuum mismatch however large it
    is.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wavelength = np.asarray(wavelength, dtype=float)
    flux = np.asarray(flux, dtype=float)
    sigma = np.sqrt(np.asarray(variance, dtype=float))

    fig, axes = plt.subplots(
        2, 1, figsize=(10, 5.5), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]}
    )
    ax = axes[0]
    ax.fill_between(
        wavelength,
        flux - sigma,
        flux + sigma,
        color="0.85",
        lw=0,
        label=r"$\pm1\sigma$ (reported variance; measured ~1.45x too small)",
    )
    ax.plot(wavelength, flux, lw=0.8, color="0.35", label="data")
    ax.plot(wavelength, fit.m0.model, lw=1.2, color="tab:blue", label="M0 continuum + narrow")
    if fit.m1 is not None:
        ax.plot(wavelength, fit.m1.model, lw=1.2, color="tab:red", label="M1 + broad")
        ax.plot(
            wavelength,
            broad_on_continuum(fit),
            lw=1.0,
            ls="--",
            color="tab:red",
            label="M1 continuum + broad",
        )
    ax.set_ylabel(r"$f_\lambda$")
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[1]
    residual_m0 = (flux - fit.m0.model) / sigma
    ax.plot(wavelength, residual_m0, lw=0.8, color="tab:blue")
    if fit.m1 is not None:
        ax.plot(wavelength, (flux - fit.m1.model) / sigma, lw=0.8, color="tab:red")
    ax.axhline(0.0, color="0.5", lw=0.8)
    ax.set_ylabel("residual / $\\sigma$")
    ax.set_xlabel(r"observed wavelength [$\AA$]")

    summary = fit.summary()
    header = (
        f"z={summary['z']:.4f}  "
        f"$\\Delta\\chi^2$={summary['delta_chi2']:.1f}  "
        f"LSF $\\sigma$={summary['lsf_sigma_angstrom']:.1f} $\\AA$"
    )
    if "broad_fwhm_kms" in summary:
        header += (
            f"  FWHM={summary['broad_fwhm_kms']:.0f} km/s"
            f"  width/LSF={summary['broad_resolution_ratio']:.1f}"
        )
    fig.suptitle(title or header, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path
