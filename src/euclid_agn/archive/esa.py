"""ESA archive backend (placeholder).

ESA is the canonical provenance source: MER/PHZ/SPE tables are served through
TAP/ADQL and spectra through IVOA DataLink at
https://easidr.esac.esa.int/tap-server/tap .  The backend is declared here so
that the interface stays honest, but it is not implemented yet: Q1 bulk work
runs against the IRSA mirror because it supports lazy byte-range FITS access.

Implementing this class is the concrete task behind the "archive parity test"
(tests/online/test_archive_parity.py), which must show that a frozen list of Q1
objects yields scientifically equivalent wavelength/flux arrays from both
backends after documented scaling.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from euclid_agn.archive.base import ArchiveBackend, SpectrumLocation

ESA_TAP_SYNC = "https://easidr.esac.esa.int/tap-server/tap/sync"
ESA_DATALINK = "https://easidr.esac.esa.int/sas-dd/data"


class EsaBackend(ArchiveBackend):
    release = "q1"

    def __init__(self, release: str = "q1") -> None:
        self.release = release

    def _not_implemented(self):
        raise NotImplementedError(
            "EsaBackend is a declared interface only. Use IrsaQ1Backend for Q1 work; "
            "implement this class when DR1 or an ESA parity test requires it."
        )

    def query_sources(self, **kwargs) -> pd.DataFrame:  # noqa: D102
        self._not_implemented()

    def query_spectrum_locations(self, **kwargs) -> list[SpectrumLocation]:  # noqa: D102
        self._not_implemented()

    def open_combined_spectrum(self, location: SpectrumLocation):  # noqa: D102
        self._not_implemented()

    def open_dither_spectra(self, location: SpectrumLocation):  # noqa: D102
        self._not_implemented()

    def get_source_context(self, object_id: int):  # noqa: D102
        self._not_implemented()

    @staticmethod
    def _unused(_: Sequence[int]) -> None:  # pragma: no cover
        return None
