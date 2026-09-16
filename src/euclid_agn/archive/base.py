"""Release-independent archive interface.

Science code must never know whether a spectrum came from the IRSA Q1 mirror or
from the ESA archive.  New releases are new backends, not new science code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from euclid_agn.spectra.types import CombinedSpectrum, DitherSpectrum, SourceContext


@dataclass(frozen=True)
class SpectrumLocation:
    """Where one object's spectrum lives inside a release."""

    object_id: int
    tile_id: int
    path: str
    hdu: int
    bandpass: str = ""
    release: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "tile_id": self.tile_id,
            "sir_path": self.path,
            "sir_hdu": self.hdu,
            "bandpass": self.bandpass,
            "release": self.release,
        }


class ArchiveBackend(ABC):
    """Abstract data access for one Euclid release."""

    release: str

    @abstractmethod
    def query_sources(
        self,
        *,
        field: str | None = None,
        tile_id: int | Sequence[int] | None = None,
        object_ids: Sequence[int] | None = None,
        limit: int | None = None,
        require_spectrum: bool = True,
    ) -> pd.DataFrame:
        """Return one row per source with catalogue context."""

    @abstractmethod
    def query_spectrum_locations(
        self,
        *,
        field: str | None = None,
        tile_id: int | Sequence[int] | None = None,
        object_ids: Sequence[int] | None = None,
        limit: int | None = None,
    ) -> list[SpectrumLocation]:
        """Return the file/HDU address of each requested spectrum."""

    @abstractmethod
    def open_combined_spectrum(self, location: SpectrumLocation) -> CombinedSpectrum:
        """Read one co-added spectrum in canonical units."""

    @abstractmethod
    def open_dither_spectra(self, location: SpectrumLocation) -> tuple[DitherSpectrum, ...]:
        """Read the individual-dither spectra contributing to one object."""

    @abstractmethod
    def get_source_context(self, object_id: int) -> SourceContext:
        """Catalogue context for one source."""
