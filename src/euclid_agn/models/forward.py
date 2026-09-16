"""Assembly of the full spectral model.

The model for one spectrum at one hypothesis is

.. math:: f(\\lambda) = C(\\lambda) + N(\\lambda) + B(\\lambda)

with a local continuum, a narrow-line system sharing one non-parametric
velocity profile, and optional broad components on permitted transitions only.
Everything is already LSF-convolved and pixel-integrated when the blocks are
built, so the comparison with the data is direct.

Two models are built per hypothesis:

``M0``  continuum + narrow
``M1``  continuum + narrow + broad

They differ by the broad block alone, so their chi-squared difference isolates
the evidence for a BLR.  Because both use the *same* narrow profile solution,
the comparison is not contaminated by the narrow component re-fitting itself
into the broad line's place - which is exactly why the narrow solve happens
first and its profile is then held fixed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from euclid_agn.fit.linear import LinearProblem, LinearSolution, delta_chi2, solve
from euclid_agn.models.broad import BroadFamily
from euclid_agn.models.narrow import NarrowFit, NarrowSystem, fit_narrow_system
from euclid_agn.numerics import blas_safe
from euclid_agn.spectra.continuum import continuum_block


@dataclass(frozen=True)
class ModelBlocks:
    """Design matrix assembled from labelled blocks."""

    design: np.ndarray
    lower: np.ndarray
    names: tuple[str, ...]
    regularisation: np.ndarray | None
    slices: dict[str, slice] = field(default_factory=dict)

    @property
    def n_parameters(self) -> int:
        return self.design.shape[1]

    def block(self, solution: LinearSolution, name: str) -> np.ndarray:
        """Coefficients of one named block from a solution."""
        return solution.coefficients[self.slices[name]]


@blas_safe
def assemble(
    wavelength: np.ndarray,
    narrow: NarrowSystem | None = None,
    narrow_profile: np.ndarray | None = None,
    broad: BroadFamily | None = None,
    n_knots: int = 8,
    degree: int = 3,
    bin_width: float | None = None,
) -> ModelBlocks:
    """Build ``[continuum | narrow | broad]`` with bounds, names and penalty.

    The narrow block needs a profile: pass the one recovered by
    :func:`euclid_agn.models.narrow.fit_narrow_system`.
    """
    wavelength = np.asarray(wavelength, dtype=np.float64)
    blocks: list[np.ndarray] = []
    lowers: list[np.ndarray] = []
    names: list[str] = []
    slices: dict[str, slice] = {}
    start = 0

    continuum_design, continuum_penalty, continuum_names = continuum_block(
        wavelength, n_knots=n_knots, degree=degree
    )
    blocks.append(continuum_design)
    lowers.append(np.full(continuum_design.shape[1], -np.inf))
    names.extend(continuum_names)
    slices["continuum"] = slice(start, start + continuum_design.shape[1])
    start += continuum_design.shape[1]

    if narrow is not None:
        if narrow_profile is None:
            raise ValueError("a narrow system needs its velocity profile")
        bases = narrow.component_bases(wavelength, bin_width)
        narrow_design = narrow.design_for_amplitudes(bases, narrow_profile)
        blocks.append(narrow_design)
        lowers.append(np.zeros(narrow_design.shape[1]))
        names.extend(f"narrow_{n}" for n in narrow.component_names)
        slices["narrow"] = slice(start, start + narrow_design.shape[1])
        start += narrow_design.shape[1]

    if broad is not None:
        broad_design = broad.design(wavelength, bin_width)
        blocks.append(broad_design)
        lowers.append(np.zeros(broad_design.shape[1]))
        names.extend(broad.names)
        slices["broad"] = slice(start, start + broad_design.shape[1])
        start += broad_design.shape[1]

    design = np.hstack(blocks)
    n_total = design.shape[1]
    regularisation = None
    if continuum_penalty.size:
        regularisation = np.zeros((continuum_penalty.shape[0], n_total))
        regularisation[:, slices["continuum"]] = continuum_penalty
    return ModelBlocks(
        design=design,
        lower=np.concatenate(lowers),
        names=tuple(names),
        regularisation=regularisation,
        slices=slices,
    )


@dataclass(frozen=True)
class HypothesisFit:
    """M0 versus M1 at one hypothesis."""

    narrow_fit: NarrowFit
    m0: LinearSolution
    m1: LinearSolution | None
    blocks_m0: ModelBlocks
    blocks_m1: ModelBlocks | None
    broad: BroadFamily | None

    @property
    def delta_chi2(self) -> float:
        """``chi2(M0) - chi2(M1)``; zero when no broad component was allowed.

        A statistic, not a significance: the broad flux is bounded at zero and
        many hypotheses are scanned per object.
        """
        if self.m1 is None:
            return 0.0
        return delta_chi2(self.m0, self.m1)

    @property
    def broad_fluxes(self) -> dict[str, float]:
        if self.m1 is None or self.blocks_m1 is None or self.broad is None:
            return {}
        values = self.blocks_m1.block(self.m1, "broad")
        return dict(zip(self.broad.line_names, (float(v) for v in values), strict=True))

    @property
    def broad_flux_errors(self) -> dict[str, float]:
        if self.m1 is None or self.blocks_m1 is None or self.broad is None:
            return {}
        errors = self.m1.errors()[self.blocks_m1.slices["broad"]]
        return dict(zip(self.broad.line_names, (float(e) for e in errors), strict=True))

    @property
    def narrow_fluxes(self) -> dict[str, float]:
        values = self.blocks_m0.block(self.m0, "narrow")
        return dict(
            zip(self.narrow_fit.system.component_names, (float(v) for v in values), strict=True)
        )

    def summary(self) -> dict[str, float | str | bool]:
        """Flat record for the screening table."""
        system = self.narrow_fit.system
        moments = self.narrow_fit.profile_moments()
        out: dict[str, float | str | bool] = {
            "z": system.z,
            "lsf_sigma_angstrom": system.lsf_sigma,
            "chi2_m0": self.m0.chi2,
            "dof_m0": self.m0.effective_dof,
            "n_pixels": float(self.m0.n_pixels),
            "delta_chi2": self.delta_chi2,
            "narrow_profile_mean_kms": moments["mean_kms"],
            "narrow_profile_sigma_kms": moments["sigma_kms"],
            "narrow_converged": self.narrow_fit.converged,
        }
        for name, value in self.narrow_fluxes.items():
            out[f"narrow_flux_{name}"] = value
        if self.m1 is not None and self.broad is not None:
            out["chi2_m1"] = self.m1.chi2
            out["dof_m1"] = self.m1.effective_dof
            out["broad_sigma_kms"] = self.broad.sigma_kms
            out["broad_velocity_kms"] = self.broad.velocity_kms
            components = self.broad.components()
            if components:
                out["broad_fwhm_kms"] = components[0].fwhm_kms
                out["broad_resolution_ratio"] = components[0].resolution_ratio
            for name, value in self.broad_fluxes.items():
                out[f"broad_flux_{name}"] = value
            for name, value in self.broad_flux_errors.items():
                out[f"broad_flux_err_{name}"] = value
        return out


@blas_safe
def fit_hypothesis(
    wavelength: np.ndarray,
    flux: np.ndarray,
    variance: np.ndarray,
    narrow: NarrowSystem,
    broad: BroadFamily | None = None,
    n_knots: int = 8,
    bin_width: float | None = None,
    smoothness: float = 1.0,
    continuum_smoothness: float = 0.0,
    max_iterations: int = 25,
) -> HypothesisFit:
    """Fit M0 and, if a broad family is supplied, M1 at one hypothesis.

    The narrow profile is solved once on M0 and then held fixed for M1, so the
    only difference between the two models is the broad block.
    """
    wavelength = np.asarray(wavelength, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    variance = np.asarray(variance, dtype=np.float64)

    continuum_design, _, _ = continuum_block(wavelength, n_knots=n_knots)
    narrow_fit = fit_narrow_system(
        wavelength,
        flux,
        variance,
        narrow,
        continuum_design=continuum_design,
        bin_width=bin_width,
        smoothness=smoothness,
        max_iterations=max_iterations,
    )

    blocks_m0 = assemble(
        wavelength,
        narrow=narrow,
        narrow_profile=narrow_fit.profile,
        broad=None,
        n_knots=n_knots,
        bin_width=bin_width,
    )
    m0 = solve(
        LinearProblem(
            design=blocks_m0.design,
            data=flux,
            variance=variance,
            lower=blocks_m0.lower,
            regularisation=blocks_m0.regularisation,
            regularisation_weight=continuum_smoothness,
            names=blocks_m0.names,
        )
    )

    m1 = None
    blocks_m1 = None
    if broad is not None:
        blocks_m1 = assemble(
            wavelength,
            narrow=narrow,
            narrow_profile=narrow_fit.profile,
            broad=broad,
            n_knots=n_knots,
            bin_width=bin_width,
        )
        m1 = solve(
            LinearProblem(
                design=blocks_m1.design,
                data=flux,
                variance=variance,
                lower=blocks_m1.lower,
                regularisation=blocks_m1.regularisation,
                regularisation_weight=continuum_smoothness,
                names=blocks_m1.names,
            )
        )

    return HypothesisFit(
        narrow_fit=narrow_fit,
        m0=m0,
        m1=m1,
        blocks_m0=blocks_m0,
        blocks_m1=blocks_m1,
        broad=broad,
    )
