"""Narrow-line system with one shared non-parametric velocity profile.

This is the component the project inherits from Chilingarian et al. (2018): all
narrow lines of a system share a single velocity profile ``P(v)``, represented
on a velocity grid rather than forced to be a Gaussian, with non-negative
amplitudes and a smoothness penalty on the profile.  The 2018 paper found that
the non-parametric representation lowered residuals and improved sensitivity to
faint broad components, which is exactly what this project needs.

Structure of the problem
------------------------
With amplitudes :math:`A_i` (one per line) and profile coefficients
:math:`p_k` (one per velocity bin), the narrow model is

.. math::  N(\\lambda) = \\sum_i A_i \\sum_k p_k \\, G(\\lambda; \\lambda_i(1+v_k/c), \\sigma_{\\rm LSF})

which is **bilinear**: linear in :math:`A` at fixed :math:`p`, and linear in
:math:`p` at fixed :math:`A`.  Each half is a bounded, regularised linear solve
of the kind :mod:`euclid_agn.fit.linear` performs, so the system is solved by
alternating between them.  Alternating minimisation on a biconvex problem
decreases the objective monotonically and converges to a stationary point; it
is not guaranteed to be the global optimum, so the solver reports its iteration
history and the caller can restart from several initial profiles.

The scale degeneracy :math:`(A, p) \\to (cA, p/c)` is fixed by normalising the
profile to unit integral over the velocity grid, so amplitudes are physical
integrated line fluxes in erg s^-1 cm^-2.

Resolution warning
------------------
At the Q1 red-grism sampling one bin is 13.4 A, and the effective LSF sigma is
about the same, so at H-alpha in the middle of the range one pixel is roughly
270 km/s.  A velocity grid much finer than that is not measuring anything: the
smoothness penalty, not the data, sets the shape between knots.  The default
grid step is therefore coarse, and :func:`suggested_velocity_step` reports the
resolution-matched value for a given line and LSF.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from euclid_agn.constants import C_KMS
from euclid_agn.fit.linear import LinearProblem, LinearSolution, solve
from euclid_agn.models.line_catalog import BY_NAME, FIXED_RATIOS, Line
from euclid_agn.numerics import blas_safe
from euclid_agn.spectra.lsf import gaussian_pixel_integral, sigma_angstrom_to_kms


def velocity_grid(half_width_kms: float = 1200.0, step_kms: float = 200.0) -> np.ndarray:
    """Symmetric velocity grid for the non-parametric profile."""
    if step_kms <= 0 or half_width_kms <= 0:
        raise ValueError("half width and step must be positive")
    n = int(np.floor(half_width_kms / step_kms))
    return np.arange(-n, n + 1, dtype=np.float64) * step_kms


def suggested_velocity_step(
    line: Line | str, z: float, lsf_sigma_angstrom: float, oversample: float = 1.0
) -> float:
    """Velocity step matching the instrumental resolution at a line.

    Anything finer is interpolation, not measurement.
    """
    line = BY_NAME[line] if isinstance(line, str) else line
    observed = line.rest * (1.0 + z)
    return sigma_angstrom_to_kms(lsf_sigma_angstrom, observed) / max(oversample, 1e-6)


@dataclass(frozen=True)
class NarrowSystem:
    """A set of narrow lines sharing one velocity profile at one redshift.

    Parameters
    ----------
    line_names : tuple of str
        Members of the system, from :mod:`euclid_agn.models.line_catalog`.
    z : float
        Systemic redshift of the hypothesis.
    lsf_sigma : float
        Effective Gaussian LSF width in Angstrom (``LSF_SIG``).
    velocities : ndarray
        Velocity grid of the shared profile, km/s.
    tie_fixed_ratios : bool
        Combine multiplets whose ratio is set by atomic physics
        (:data:`euclid_agn.models.line_catalog.FIXED_RATIOS`) into one free
        amplitude.  Density-sensitive ratios such as [S II] 6716/6731 stay free.
    velocity_offset : float
        Rigid velocity shift of the whole system, km/s.  Part of the *non-linear*
        parameter set; the linear solves see it as fixed.
    """

    line_names: tuple[str, ...]
    z: float
    lsf_sigma: float
    velocities: np.ndarray = field(default_factory=velocity_grid)
    tie_fixed_ratios: bool = True
    velocity_offset: float = 0.0

    def __post_init__(self) -> None:
        if not self.line_names:
            raise ValueError("a narrow system needs at least one line")
        unknown = [n for n in self.line_names if n not in BY_NAME]
        if unknown:
            raise ValueError(f"unknown lines: {unknown}")
        if self.lsf_sigma <= 0:
            raise ValueError("lsf_sigma must be positive")
        object.__setattr__(self, "velocities", np.asarray(self.velocities, dtype=np.float64))

    # -- bookkeeping ------------------------------------------------------
    @property
    def n_velocities(self) -> int:
        return int(self.velocities.size)

    def components(self) -> tuple[tuple[str, tuple[tuple[str, float], ...]], ...]:
        """Free amplitudes and the lines each one drives.

        Returns a tuple of ``(name, ((line_name, relative_strength), ...))``.
        With ``tie_fixed_ratios`` a multiplet appears once, carrying its weak
        partner at the fixed ratio.
        """
        members = list(self.line_names)
        tied: dict[str, list[tuple[str, float]]] = {}
        consumed: set[str] = set()
        if self.tie_fixed_ratios:
            for (strong, weak), ratio in FIXED_RATIOS.items():
                if strong in members and weak in members:
                    tied[strong] = [(strong, 1.0), (weak, ratio)]
                    consumed.add(weak)
        out = []
        for name in members:
            if name in consumed:
                continue
            out.append((name, tuple(tied.get(name, [(name, 1.0)]))))
        return tuple(out)

    @property
    def component_names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.components())

    def observed_wavelength(self, line_name: str, velocity_kms: float = 0.0) -> float:
        rest = BY_NAME[line_name].rest
        total = self.velocity_offset + velocity_kms
        return rest * (1.0 + self.z) * (1.0 + total / C_KMS)

    # -- bases ------------------------------------------------------------
    def velocity_basis(
        self, wavelength: np.ndarray, line_name: str, bin_width: float | None = None
    ) -> np.ndarray:
        """``(n_pixels, n_velocities)`` basis for one line.

        Column ``k`` is a unit-flux LSF-broadened line placed at velocity
        ``velocities[k]`` relative to the systemic redshift.
        """
        wavelength = np.asarray(wavelength, dtype=np.float64)
        out = np.empty((wavelength.size, self.n_velocities))
        for k, v in enumerate(self.velocities):
            centre = self.observed_wavelength(line_name, v)
            out[:, k] = gaussian_pixel_integral(wavelength, centre, self.lsf_sigma, bin_width)
        return out

    def component_bases(
        self, wavelength: np.ndarray, bin_width: float | None = None
    ) -> dict[str, np.ndarray]:
        """Velocity basis per free amplitude, multiplets already combined."""
        bases: dict[str, np.ndarray] = {}
        for name, members in self.components():
            block = None
            for line_name, strength in members:
                contribution = strength * self.velocity_basis(wavelength, line_name, bin_width)
                block = contribution if block is None else block + contribution
            bases[name] = block
        return bases

    @blas_safe
    def design_for_amplitudes(
        self, bases: dict[str, np.ndarray], profile: np.ndarray
    ) -> np.ndarray:
        """``(n_pixels, n_components)`` design at a fixed profile."""
        profile = np.asarray(profile, dtype=np.float64)
        return np.column_stack([bases[name] @ profile for name in self.component_names])

    @blas_safe
    def design_for_profile(
        self, bases: dict[str, np.ndarray], amplitudes: np.ndarray
    ) -> np.ndarray:
        """``(n_pixels, n_velocities)`` design at fixed amplitudes."""
        amplitudes = np.asarray(amplitudes, dtype=np.float64)
        total = None
        for amplitude, name in zip(amplitudes, self.component_names, strict=True):
            block = amplitude * bases[name]
            total = block if total is None else total + block
        return total

    @blas_safe
    def evaluate(
        self,
        wavelength: np.ndarray,
        amplitudes: np.ndarray,
        profile: np.ndarray,
        bin_width: float | None = None,
    ) -> np.ndarray:
        """Narrow-line model flux density."""
        bases = self.component_bases(wavelength, bin_width)
        return self.design_for_profile(bases, amplitudes) @ np.asarray(profile, dtype=float)


def smoothness_penalty(n_velocities: int, order: int = 2) -> np.ndarray:
    """Second-difference operator on the profile coefficients.

    Structure only; the strength is the dimensionless ``smoothness`` passed to
    :func:`fit_narrow_system`.
    """
    if n_velocities <= order:
        return np.zeros((0, n_velocities))
    d = np.eye(n_velocities)
    for _ in range(order):
        d = np.diff(d, axis=0)
    return d


def gaussian_profile(velocities: np.ndarray, sigma_kms: float = 250.0) -> np.ndarray:
    """Unit-integral Gaussian starting profile."""
    velocities = np.asarray(velocities, dtype=np.float64)
    p = np.exp(-0.5 * (velocities / sigma_kms) ** 2)
    return normalise_profile(p, velocities)


def normalise_profile(profile: np.ndarray, velocities: np.ndarray) -> np.ndarray:
    """Scale a profile to unit integral over the velocity grid."""
    profile = np.asarray(profile, dtype=np.float64)
    step = float(np.median(np.diff(velocities))) if velocities.size > 1 else 1.0
    total = float(profile.sum() * step)
    if total <= 0:
        raise ValueError("profile has non-positive integral and cannot be normalised")
    return profile / total


@dataclass(frozen=True)
class NarrowFit:
    """Result of the alternating narrow-line solve."""

    system: NarrowSystem
    amplitudes: np.ndarray
    profile: np.ndarray
    continuum: np.ndarray
    solution: LinearSolution
    chi2_history: tuple[float, ...]
    converged: bool

    @property
    def chi2(self) -> float:
        return self.solution.chi2

    def amplitude(self, name: str) -> float:
        return float(self.amplitudes[self.system.component_names.index(name)])

    def profile_moments(self) -> dict[str, float]:
        """Mean velocity and velocity dispersion of the recovered profile."""
        v = self.system.velocities
        p = np.clip(self.profile, 0.0, None)
        total = p.sum()
        if total <= 0:
            return {"mean_kms": np.nan, "sigma_kms": np.nan}
        mean = float((p * v).sum() / total)
        variance = float((p * (v - mean) ** 2).sum() / total)
        return {"mean_kms": mean, "sigma_kms": float(np.sqrt(max(variance, 0.0)))}


@blas_safe
def fit_narrow_system(
    wavelength: np.ndarray,
    flux: np.ndarray,
    variance: np.ndarray,
    system: NarrowSystem,
    continuum_design: np.ndarray | None = None,
    bin_width: float | None = None,
    smoothness: float = 1.0,
    initial_profile: np.ndarray | None = None,
    max_iterations: int = 25,
    tolerance: float = 1e-6,
) -> NarrowFit:
    """Alternating constrained solve for amplitudes and the shared profile.

    Continuum coefficients are solved jointly in both half-steps, so the
    continuum never absorbs line flux at one step and give it back at the next.

    Parameters
    ----------
    continuum_design : ndarray, optional
        ``(n_pixels, n_continuum)`` block; use
        :func:`euclid_agn.spectra.continuum.continuum_block`.  ``None`` fits no
        continuum, which is only sensible on continuum-subtracted input.
    smoothness : float
        Weight of the second-difference penalty on the profile, applied
        relative to the data term (see
        :func:`euclid_agn.fit.linear.solve`).
    """
    wavelength = np.asarray(wavelength, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    variance = np.asarray(variance, dtype=np.float64)
    bases = system.component_bases(wavelength, bin_width)
    n_components = len(system.component_names)
    n_continuum = 0 if continuum_design is None else continuum_design.shape[1]

    profile = (
        normalise_profile(np.asarray(initial_profile, dtype=float), system.velocities)
        if initial_profile is not None
        else gaussian_profile(system.velocities)
    )
    amplitudes = np.zeros(n_components)
    continuum = np.zeros(n_continuum)
    history: list[float] = []
    solution: LinearSolution | None = None
    converged = False

    for _ in range(max_iterations):
        # --- step A: amplitudes (and continuum) at fixed profile ---
        line_design = system.design_for_amplitudes(bases, profile)
        design = _stack(continuum_design, line_design)
        lower = np.concatenate([np.full(n_continuum, -np.inf), np.zeros(n_components)])
        names = tuple(f"continuum_{i}" for i in range(n_continuum)) + system.component_names
        solution = solve(
            LinearProblem(
                design=design, data=flux, variance=variance, lower=lower, names=names
            )
        )
        continuum = solution.coefficients[:n_continuum]
        amplitudes = solution.coefficients[n_continuum:]

        if not np.any(amplitudes > 0):
            history.append(solution.chi2)
            break

        # --- step B: profile (and continuum) at fixed amplitudes ---
        profile_design = system.design_for_profile(bases, amplitudes)
        design = _stack(continuum_design, profile_design)
        n_velocity = system.n_velocities
        lower = np.concatenate([np.full(n_continuum, -np.inf), np.zeros(n_velocity)])
        penalty = smoothness_penalty(n_velocity)
        regularisation = (
            np.hstack([np.zeros((penalty.shape[0], n_continuum)), penalty])
            if penalty.size
            else None
        )
        names = tuple(f"continuum_{i}" for i in range(n_continuum)) + tuple(
            f"profile_{k}" for k in range(n_velocity)
        )
        solution = solve(
            LinearProblem(
                design=design,
                data=flux,
                variance=variance,
                lower=lower,
                regularisation=regularisation,
                regularisation_weight=smoothness,
                names=names,
            )
        )
        continuum = solution.coefficients[:n_continuum]
        raw_profile = solution.coefficients[n_continuum:]
        if raw_profile.sum() <= 0:
            history.append(solution.chi2)
            break
        # Move the scale out of the profile and into the amplitudes.
        step = float(np.median(np.diff(system.velocities))) if n_velocity > 1 else 1.0
        scale = float(raw_profile.sum() * step)
        profile = raw_profile / scale
        amplitudes = amplitudes * scale

        history.append(solution.chi2)
        if len(history) >= 2 and abs(history[-2] - history[-1]) <= tolerance * max(
            abs(history[-1]), 1.0
        ):
            converged = True
            break

    assert solution is not None
    return NarrowFit(
        system=system,
        amplitudes=amplitudes,
        profile=profile,
        continuum=continuum,
        solution=solution,
        chi2_history=tuple(history),
        converged=converged,
    )


def _stack(continuum_design: np.ndarray | None, block: np.ndarray) -> np.ndarray:
    if continuum_design is None:
        return block
    return np.hstack([continuum_design, block])
