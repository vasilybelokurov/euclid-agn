"""Typed configuration loaded from YAML.

Configuration is hashed into the run manifest, so every number that can change
a scientific result must live here rather than in a function default.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


class ArchiveConfig(BaseModel):
    backend: Literal["irsa", "esa"] = "irsa"
    release: str = "q1"
    anon_s3: bool = True
    tap_timeout_s: float = 900.0
    cache_dir: Path | None = None


class SelectionConfig(BaseModel):
    """Which spectra enter the run.  Data availability only - never host properties."""

    field: str | None = None
    tile_ids: list[int] = Field(default_factory=list)
    object_ids: list[int] = Field(default_factory=list)
    limit: int | None = None
    min_usable_pixel_fraction: float = 0.5
    wavelength_min: float = 12500.0
    wavelength_max: float = 18500.0
    reject_mask_bits: list[str] = Field(default_factory=lambda: ["NOT_USE"])


class ContinuumConfig(BaseModel):
    kind: Literal["bspline"] = "bspline"
    n_knots: int = 8
    degree: int = 3
    smoothness: float = 0.0


class NarrowLineConfig(BaseModel):
    """Non-parametric shared NLR velocity profile."""

    velocity_half_width_kms: float = 1200.0
    velocity_step_kms: float = 100.0
    smoothness_lambda: float = 1.0
    non_negative: bool = True


class BroadLineConfig(BaseModel):
    sigma_min_kms: float = 300.0
    sigma_max_kms: float = 6000.0
    sigma_grid_n: int = 12
    velocity_offset_max_kms: float = 1500.0
    allowed_transitions: list[str] = Field(
        default_factory=lambda: ["Halpha", "Hbeta", "Pabeta", "Pagamma", "HeI10830", "MgII"]
    )


class ScreeningConfig(BaseModel):
    redshift_sources: list[str] = Field(
        default_factory=lambda: ["spe", "phz", "peaks", "blind"]
    )
    blind_z_min: float = 0.0
    blind_z_max: float = 5.7
    blind_z_step_kms: float = 300.0
    noise_scale_free: bool = True


class ValidationConfig(BaseModel):
    n_injections: int = 1000
    seed: int = 20260916
    broad_flux_dex_range: tuple[float, float] = (-18.0, -15.0)
    broad_fwhm_kms_range: tuple[float, float] = (800.0, 8000.0)
    redshift_range: tuple[float, float] = (0.9, 1.8)


class Config(BaseModel):
    """Top-level configuration object."""

    name: str = "default"
    archive: ArchiveConfig = Field(default_factory=ArchiveConfig)
    selection: SelectionConfig = Field(default_factory=SelectionConfig)
    continuum: ContinuumConfig = Field(default_factory=ContinuumConfig)
    narrow: NarrowLineConfig = Field(default_factory=NarrowLineConfig)
    broad: BroadLineConfig = Field(default_factory=BroadLineConfig)
    screening: ScreeningConfig = Field(default_factory=ScreeningConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    output_dir: Path = Path("outputs")
    seed: int = 20260916

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        with open(path) as fh:
            payload = yaml.safe_load(fh) or {}
        return cls.model_validate(payload)

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self.model_dump_json())

    def checksum(self) -> str:
        """SHA-256 of the canonical JSON form; recorded in the run manifest."""
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()
