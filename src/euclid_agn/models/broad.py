"""Broad-line (BLR) components.

Only *permitted* transitions may carry a broad component in the AGN model.  The
fitter is able to attach one to a forbidden line, and
:func:`forbidden_broad_basis` exists precisely so that null tests can do it, but
:class:`BroadComponent` refuses by default: a broad [O III] is a pipeline
diagnostic, never evidence for a BLR.

Parameterisation follows the linear/non-linear split used throughout: the width
``sigma_kms`` and velocity offset ``velocity_kms`` are non-linear parameters
scanned or optimised outside the linear solve, while the integrated flux is a
linear, non-negative coefficient.

No universal minimum FWHM is imposed.  What counts as "broad" in Euclid depends
on the effective LSF of the individual source, which varies by a factor of eight
across Q1, so the pipeline records the intrinsic width and its ratio to the
instrumental width and lets the selection function speak.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from euclid_agn.constants import C_KMS
from euclid_agn.models.line_catalog import BY_NAME, Line
from euclid_agn.spectra.lsf import (
    FWHM_OVER_SIGMA,
    effective_sigma,
    gaussian_pixel_integral,
    pixel_edges,
    sigma_angstrom_to_kms,
    sigma_kms_to_angstrom,
)


@dataclass(frozen=True)
class BroadComponent:
    """One broad Gaussian attached to a permitted transition.

    Parameters
    ----------
    line_name : str
        Transition carrying the component; must be permitted.
    z : float
        Systemic redshift.
    sigma_kms : float
        Intrinsic velocity dispersion, before the instrumental LSF.
    velocity_kms : float
        Offset of the broad component from systemic.
    lsf_sigma : float
        Effective instrumental width, Angstrom.
    allow_forbidden : bool
        Escape hatch for null tests only.  Leave false in science runs.
    """

    line_name: str
    z: float
    sigma_kms: float
    lsf_sigma: float
    velocity_kms: float = 0.0
    allow_forbidden: bool = False

    def __post_init__(self) -> None:
        if self.line_name not in BY_NAME:
            raise ValueError(f"unknown line {self.line_name!r}")
        if not self.line.permitted and not self.allow_forbidden:
            raise ValueError(
                f"{self.line_name} is a forbidden transition and cannot carry a BLR "
                "component; pass allow_forbidden=True only for a null test"
            )
        if self.sigma_kms <= 0:
            raise ValueError("sigma_kms must be positive")
        if self.lsf_sigma <= 0:
            raise ValueError("lsf_sigma must be positive")

    @property
    def line(self) -> Line:
        return BY_NAME[self.line_name]

    @property
    def centre(self) -> float:
        """Observed central wavelength, Angstrom."""
        return self.line.rest * (1.0 + self.z) * (1.0 + self.velocity_kms / C_KMS)

    @property
    def intrinsic_sigma_angstrom(self) -> float:
        return sigma_kms_to_angstrom(self.sigma_kms, self.centre)

    @property
    def observed_sigma_angstrom(self) -> float:
        """Width actually seen: intrinsic and instrumental in quadrature."""
        return effective_sigma(self.intrinsic_sigma_angstrom, self.lsf_sigma)

    @property
    def fwhm_kms(self) -> float:
        """Intrinsic FWHM, the number quoted for BLR widths."""
        return FWHM_OVER_SIGMA * self.sigma_kms

    @property
    def resolution_ratio(self) -> float:
        """Intrinsic width divided by the instrumental width at this line.

        Below about 1 the component is barely distinguishable from an
        unresolved line and any recovered width is dominated by the LSF model.
        This ratio is a primary axis of the selection function.
        """
        return self.intrinsic_sigma_angstrom / self.lsf_sigma

    @property
    def lsf_sigma_kms(self) -> float:
        return sigma_angstrom_to_kms(self.lsf_sigma, self.centre)

    def basis(self, wavelength: np.ndarray, bin_width: float | None = None) -> np.ndarray:
        """Unit-flux, LSF-convolved, pixel-integrated profile."""
        return gaussian_pixel_integral(
            np.asarray(wavelength, dtype=np.float64),
            self.centre,
            self.observed_sigma_angstrom,
            bin_width,
        )

    def in_range(self, wavelength: np.ndarray, n_sigma: float = 2.0) -> bool:
        """Is the component's core inside the covered wavelength range?

        A component whose centre lies outside the data can still change chi2 by
        its wings, which is a classic way to manufacture a spurious detection.
        """
        wavelength = np.asarray(wavelength)
        if wavelength.size == 0:
            return False
        reach = n_sigma * self.observed_sigma_angstrom
        return bool(
            (self.centre >= wavelength[0] - reach) and (self.centre <= wavelength[-1] + reach)
        )

    def contained_fraction(self, wavelength: np.ndarray, bin_width: float | None = None) -> float:
        """Fraction of the line's flux that falls on covered pixels.

        A broad component near an edge is mostly outside the data, so its
        amplitude is set by a truncated wing and is free to absorb whatever
        edge artefact happens to be there.  The first real-data pilot produced
        exactly that: its best candidate was a broad profile centred at
        18200 Angstrom on the red edge of the grism.  Requiring most of the
        profile to be observed removes the failure without removing genuine
        lines, which sit well inside the range.
        """
        wavelength = np.asarray(wavelength, dtype=np.float64)
        if wavelength.size == 0:
            return 0.0
        profile = self.basis(wavelength, bin_width)
        widths = np.diff(pixel_edges(wavelength, bin_width))
        return float(np.sum(profile * widths))


@dataclass(frozen=True)
class BroadFamily:
    """Several permitted lines sharing one broad width and velocity offset.

    Balmer lines of the same object should not be given independent BLR widths
    without evidence; sharing them is the default and each line keeps its own
    non-negative flux.
    """

    line_names: tuple[str, ...]
    z: float
    sigma_kms: float
    lsf_sigma: float
    velocity_kms: float = 0.0
    allow_forbidden: bool = False

    def components(self) -> tuple[BroadComponent, ...]:
        return tuple(
            BroadComponent(
                line_name=name,
                z=self.z,
                sigma_kms=self.sigma_kms,
                lsf_sigma=self.lsf_sigma,
                velocity_kms=self.velocity_kms,
                allow_forbidden=self.allow_forbidden,
            )
            for name in self.line_names
        )

    def design(self, wavelength: np.ndarray, bin_width: float | None = None) -> np.ndarray:
        """``(n_pixels, n_lines)`` design block, one column per broad line."""
        return np.column_stack(
            [component.basis(wavelength, bin_width) for component in self.components()]
        )

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(f"broad_{name}" for name in self.line_names)

    def visible_components(
        self, wavelength: np.ndarray, n_sigma: float = 2.0
    ) -> tuple[BroadComponent, ...]:
        return tuple(c for c in self.components() if c.in_range(wavelength, n_sigma))


def sigma_grid(
    sigma_min_kms: float = 300.0, sigma_max_kms: float = 6000.0, n: int = 12
) -> np.ndarray:
    """Logarithmic grid of trial broad-line widths.

    Logarithmic because the interesting range spans more than a decade and the
    chi2 surface is much flatter at large widths.
    """
    if sigma_min_kms <= 0 or sigma_max_kms <= sigma_min_kms or n < 2:
        raise ValueError("need 0 < sigma_min < sigma_max and n >= 2")
    return np.geomspace(sigma_min_kms, sigma_max_kms, n)


def forbidden_broad_basis(
    line_name: str,
    wavelength: np.ndarray,
    z: float,
    sigma_kms: float,
    lsf_sigma: float,
    bin_width: float | None = None,
) -> np.ndarray:
    """Broad profile on a forbidden transition, for null tests only.

    The machinery will happily fit this.  The AGN model must never read a
    detection here as BLR emission; a positive result means the pipeline is
    responding to continuum mismatch, contamination or noise.
    """
    component = BroadComponent(
        line_name=line_name,
        z=z,
        sigma_kms=sigma_kms,
        lsf_sigma=lsf_sigma,
        allow_forbidden=True,
    )
    return component.basis(wavelength, bin_width)
