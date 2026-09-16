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
    ax.fill_between(w, ylim[0], ylim[1], where=~ok, color="0.88", lw=0, step="mid", zorder=0)
    ax.axvspan(w[0] - 200.0, RGS_SCIENCE_WMIN_ANGSTROM, color="0.96", lw=0, zorder=-1)
    ax.axvspan(RGS_SCIENCE_WMAX_ANGSTROM, w[-1] + 200.0, color="0.96", lw=0, zorder=-1)
    if show_sigma:
        ax.fill_between(w, f - sigma, f + sigma, alpha=0.3, lw=0, color="tab:blue", zorder=2)
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
    axes[-1].set_xlabel(r"observed wavelength [$\AA$]  (shaded: outside 12500-18500 $\AA$)")
    fig.suptitle(
        title or f"object {observation.object_id}   tile {observation.source.tile_id}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path
