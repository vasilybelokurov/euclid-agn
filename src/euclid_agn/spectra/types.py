"""Immutable domain objects for Euclid spectra.

All arrays are plain NumPy in canonical units (Angstrom; erg/s/cm2/Angstrom).
Units are validated at the IO boundary, not in inner loops.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from euclid_agn.spectra.masking import MaskDefinition, mask_bit_fractions, usable_pixels


def _frozen(a: np.ndarray, dtype) -> np.ndarray:
    out = np.asarray(a, dtype=dtype)
    out = out.copy()
    out.setflags(write=False)
    return out


@dataclass(frozen=True)
class Spectrum1D:
    """One extracted 1D spectrum on the SIR wavelength grid.

    Parameters
    ----------
    wavelength : ndarray
        Bin-centre wavelength, Angstrom, strictly increasing.
    flux : ndarray
        Flux density, erg s^-1 cm^-2 Angstrom^-1, FSCALE already applied.
    variance : ndarray
        Variance of ``flux``, (erg s^-1 cm^-2 Angstrom^-1)^2, FSCALE^2 applied.
    mask : ndarray of int
        SIR MASK column (bit field).
    quality : ndarray
        SIR QUALITY column (float, 0-1 in Q1 products).
    lsf_sigma : float
        Effective Gaussian LSF standard deviation in Angstrom (``LSF_SIG``).
    bin_width : float
        Wavelength bin width in Angstrom (``BINWIDTH``).
    metadata : dict
        Provenance and header keywords; never used for science decisions
        without being copied into an explicit field first.
    """

    wavelength: np.ndarray
    flux: np.ndarray
    variance: np.ndarray
    mask: np.ndarray
    quality: np.ndarray
    lsf_sigma: float
    bin_width: float
    mask_definition: MaskDefinition = field(default_factory=MaskDefinition.default)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.wavelength)
        for name in ("flux", "variance", "mask", "quality"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"{name} has length {len(getattr(self, name))}, expected {n}")
        object.__setattr__(self, "wavelength", _frozen(self.wavelength, np.float64))
        object.__setattr__(self, "flux", _frozen(self.flux, np.float64))
        object.__setattr__(self, "variance", _frozen(self.variance, np.float64))
        object.__setattr__(self, "mask", _frozen(self.mask, np.int64))
        object.__setattr__(self, "quality", _frozen(self.quality, np.float64))
        if n > 1 and not np.all(np.diff(self.wavelength) > 0):
            raise ValueError("wavelength must be strictly increasing")

    @property
    def n_pixels(self) -> int:
        return len(self.wavelength)

    def with_variance_scale(self, factor: float):
        """A copy with the variance multiplied by ``factor``.

        This is how the measured per-object noise scale enters the likelihood:
        the archive variance is left untouched, a derived spectrum carries the
        corrected one, and ``metadata["variance_scale"]`` records the factor.
        """
        if not np.isfinite(factor) or factor <= 0:
            raise ValueError("variance scale must be a positive finite number")
        metadata = dict(self.metadata)
        metadata["variance_scale"] = float(factor) * float(metadata.get("variance_scale", 1.0))
        return replace(self, variance=self.variance * float(factor), metadata=metadata)

    def with_extra_mask(self, bad: np.ndarray, reason: str = "coherence"):
        """A copy with ``NOT_USE`` set on ``bad`` pixels; the archive mask is never edited in place."""
        bad = np.asarray(bad, dtype=bool)
        if bad.shape != self.mask.shape:
            raise ValueError("extra mask must match the wavelength grid")
        metadata = dict(self.metadata)
        metadata[f"extra_mask_{reason}"] = int(bad.sum())
        return replace(self, mask=np.where(bad, self.mask | 1, self.mask), metadata=metadata)

    def usable(self, reject: tuple[str, ...] | None = None) -> np.ndarray:
        """Boolean array of pixels admissible to a fit.

        A pixel is usable when no rejected mask bit is set, the variance is
        positive and finite, and the flux is finite.  Masked pixels are never
        interpolated over; they are dropped from the likelihood.
        """
        kwargs = {} if reject is None else {"reject": reject}
        ok = usable_pixels(self.mask, self.mask_definition, **kwargs)
        ok &= np.isfinite(self.flux)
        ok &= np.isfinite(self.variance) & (self.variance > 0)
        return ok

    def quality_metrics(self) -> dict[str, float]:
        """Scalar diagnostics recorded for every spectrum (selection-function inputs)."""
        ok = self.usable()
        n = self.n_pixels
        out: dict[str, float] = {
            "n_pixels": float(n),
            "usable_pixel_fraction": float(np.count_nonzero(ok)) / n if n else float("nan"),
            "median_quality": float(np.median(self.quality)) if n else float("nan"),
            "lsf_sigma_angstrom": float(self.lsf_sigma),
        }
        if np.any(ok):
            out["wavelength_min_usable"] = float(self.wavelength[ok].min())
            out["wavelength_max_usable"] = float(self.wavelength[ok].max())
            with np.errstate(invalid="ignore", divide="ignore"):
                snr = self.flux[ok] / np.sqrt(self.variance[ok])
            out["median_snr_per_pixel"] = float(np.median(snr))
        else:
            out["wavelength_min_usable"] = float("nan")
            out["wavelength_max_usable"] = float("nan")
            out["median_snr_per_pixel"] = float("nan")
        for name, frac in mask_bit_fractions(self.mask, self.mask_definition).items():
            out[f"mask_frac_{name.lower()}"] = frac
        return out


@dataclass(frozen=True)
class DitherSpectrum(Spectrum1D):
    """A single-dither extraction (``*_DITH1D_<PTGID>_SIGNAL``)."""

    dither_id: int = -1
    pointing_id: int = -1
    detector_id: int = -1
    gwa_position: str = ""
    gwa_tilt: float = float("nan")
    exposure_time: float = float("nan")
    extraction_profile: str = ""
    contaminant_object_ids: tuple[int, ...] = ()

    @property
    def n_contaminants(self) -> int:
        return len(self.contaminant_object_ids)


@dataclass(frozen=True)
class CombinedSpectrum(Spectrum1D):
    """The co-added spectrum (``*_COMBINED1D_SIGNAL``)."""

    ndith: np.ndarray | None = None
    exposure_time: float = float("nan")
    extraction_profile: str = ""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.ndith is not None:
            if len(self.ndith) != self.n_pixels:
                raise ValueError("ndith length must match the wavelength grid")
            object.__setattr__(self, "ndith", _frozen(self.ndith, np.int32))

    def quality_metrics(self) -> dict[str, float]:
        out = super().quality_metrics()
        if self.ndith is not None:
            ok = self.usable()
            out["ndith_max"] = float(np.max(self.ndith)) if self.n_pixels else float("nan")
            out["ndith_median_usable"] = (
                float(np.median(self.ndith[ok])) if np.any(ok) else float("nan")
            )
        return out


@dataclass(frozen=True)
class SourceContext:
    """Catalogue context for one source.  Never an AGN-selection input."""

    object_id: int
    ra: float
    dec: float
    tile_id: int
    release: str = "q1"
    photometry: dict[str, float] = field(default_factory=dict)
    morphology: dict[str, float] = field(default_factory=dict)
    photometric_redshift: dict[str, float] = field(default_factory=dict)
    spectroscopic_hypotheses: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class SpectralObservation:
    """Everything the pipeline knows about one observed source."""

    source: SourceContext
    combined: CombinedSpectrum
    dithers: tuple[DitherSpectrum, ...] = ()

    @property
    def object_id(self) -> int:
        return self.source.object_id

    @property
    def n_dithers(self) -> int:
        return len(self.dithers)

    def contamination_metrics(self) -> dict[str, float]:
        """Contamination bookkeeping across dithers."""
        if not self.dithers:
            return {
                "n_dithers": 0.0,
                "contaminants_total": float("nan"),
                "contaminants_max_per_dither": float("nan"),
                "contaminants_unique": float("nan"),
            }
        per = [d.n_contaminants for d in self.dithers]
        unique: set[int] = set()
        for d in self.dithers:
            unique.update(d.contaminant_object_ids)
        return {
            "n_dithers": float(len(self.dithers)),
            "contaminants_total": float(sum(per)),
            "contaminants_max_per_dither": float(max(per)),
            "contaminants_unique": float(len(unique)),
        }
