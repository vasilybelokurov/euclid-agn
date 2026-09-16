"""Reader for ``DpdSirCombinedSpectra`` FITS products.

File layout VERIFIED against
``q1/SIR/102160339/EUC_SIR_W-COMBSPEC_102160339_2024-11-05T16:26:34.614296Z.fits``
(112 MB, 9949 HDUs, ``N_OBJ = 1000``)::

    PRIMARY                             FITS_DEF='sir.combinedSpectra', TILE_ID, N_OBJ,
                                        LRANGE, MSK_FLAG_* bit definitions
    <k>_META                            empty IMAGE HDU; OBJ_ID, RA_OBJ, DEC_OBJ, N_DITH
    <k>_COMBINED1D_SIGNAL               BINTABLE WAVELENGTH/SIGNAL/MASK/QUALITY/VAR/NDITH
    <k>_DITH1D_<PTGID>_SIGNAL           BINTABLE WAVELENGTH/SIGNAL/MASK/QUALITY/VAR
    <k>_DITH1D_<PTGID>_CONTAMINANTS     BINTABLE OBJ_ID of overlapping sources
    ... repeated per dither, then the next object ...

``<k>`` is the index of the object *within the file*, not its OBJ_ID, and the
number of HDUs per object varies with the number of contributing dithers, so
groups must be discovered from EXTNAME rather than assumed.

Flux scaling
------------
``SIGNAL`` and ``VAR`` are stored in units of ``FSCALE``.  The DPDD states that
FSCALE applies to SIGNAL and must also be accounted for in VAR; the quadratic
convention (``VAR_phys = VAR * FSCALE**2``) was confirmed empirically on 973
real spectra by comparing the pixel-to-pixel scatter of SIGNAL against
``sqrt(VAR)`` in the raw stored units (median ratio 1.25, 5th percentile 0.99).
A ratio of 1e-16 or 1e16 would have indicated the alternative convention.
See :func:`euclid_agn.io.sir.variance_scale` and the regression test
``tests/regression/test_fscale_convention.py``.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

from euclid_agn.spectra.masking import MaskDefinition
from euclid_agn.spectra.types import CombinedSpectrum, DitherSpectrum, SourceContext

_META_RE = re.compile(r"^(\d+)_META$")
_COMBINED_RE = re.compile(r"^(\d+)_COMBINED1D_SIGNAL$")
_DITHER_SIGNAL_RE = re.compile(r"^(\d+)_DITH1D_(\d+)_SIGNAL$")
_DITHER_CONTAM_RE = re.compile(r"^(\d+)_DITH1D_(\d+)_CONTAMINANTS$")


def flux_scale(header) -> float:
    """FSCALE for a SIGNAL table; 1.0 if the keyword is absent."""
    return float(header.get("FSCALE", 1.0))


def variance_scale(header) -> float:
    """Multiplicative factor taking stored VAR to physical variance."""
    return flux_scale(header) ** 2


@dataclass(frozen=True)
class ObjectHduGroup:
    """HDU indices belonging to one object inside a SIR file."""

    object_index: int
    object_id: int
    meta_hdu: int
    combined_hdu: int
    dither_signal_hdus: dict[int, int]  # pointing_id -> hdu index
    dither_contam_hdus: dict[int, int]  # pointing_id -> hdu index

    @property
    def n_dither_hdus(self) -> int:
        return len(self.dither_signal_hdus)


class SirCombinedSpectraFile:
    """Lazy accessor to one SIR combined-spectra FITS file.

    The FITS file is treated as immutable source data: nothing is written back.

    Examples
    --------
    >>> with open_sir_file(path) as sir:            # doctest: +SKIP
    ...     obs = sir.read_observation(2731173428682078045)
    """

    def __init__(self, hdulist: fits.HDUList, origin: str = "") -> None:
        self._hdul = hdulist
        self.origin = origin
        self._extnames: list[str] = [h.name for h in hdulist]
        self._groups: dict[int, ObjectHduGroup] | None = None
        self._by_object_id: dict[int, int] | None = None

    # -- file level -------------------------------------------------------
    @property
    def primary_header(self):
        return self._hdul[0].header

    @property
    def tile_id(self) -> int:
        return int(self.primary_header["TILE_ID"])

    @property
    def n_objects_header(self) -> int:
        return int(self.primary_header["N_OBJ"])

    @property
    def grism_combination(self) -> str:
        return str(self.primary_header.get("LRANGE", "")).strip()

    @property
    def mask_definition(self) -> MaskDefinition:
        return MaskDefinition.from_header(self.primary_header)

    # -- HDU discovery ----------------------------------------------------
    def groups(self) -> dict[int, ObjectHduGroup]:
        """Discover per-object HDU groups from EXTNAMEs (header scan only)."""
        if self._groups is not None:
            return self._groups
        meta: dict[int, int] = {}
        combined: dict[int, int] = {}
        dsig: dict[int, dict[int, int]] = {}
        dcon: dict[int, dict[int, int]] = {}
        for i, name in enumerate(self._extnames):
            if (m := _META_RE.match(name)) is not None:
                meta[int(m.group(1))] = i
            elif (m := _COMBINED_RE.match(name)) is not None:
                combined[int(m.group(1))] = i
            elif (m := _DITHER_SIGNAL_RE.match(name)) is not None:
                dsig.setdefault(int(m.group(1)), {})[int(m.group(2))] = i
            elif (m := _DITHER_CONTAM_RE.match(name)) is not None:
                dcon.setdefault(int(m.group(1)), {})[int(m.group(2))] = i
        groups: dict[int, ObjectHduGroup] = {}
        by_id: dict[int, int] = {}
        for idx, meta_hdu in sorted(meta.items()):
            if idx not in combined:
                continue
            object_id = int(self._hdul[meta_hdu].header["OBJ_ID"])
            groups[idx] = ObjectHduGroup(
                object_index=idx,
                object_id=object_id,
                meta_hdu=meta_hdu,
                combined_hdu=combined[idx],
                dither_signal_hdus=dict(sorted(dsig.get(idx, {}).items())),
                dither_contam_hdus=dict(sorted(dcon.get(idx, {}).items())),
            )
            by_id[object_id] = idx
        self._groups = groups
        self._by_object_id = by_id
        return groups

    def object_ids(self) -> list[int]:
        self.groups()
        assert self._by_object_id is not None
        return sorted(self._by_object_id)

    def group_for_object(self, object_id: int) -> ObjectHduGroup:
        self.groups()
        assert self._by_object_id is not None and self._groups is not None
        if object_id not in self._by_object_id:
            raise KeyError(f"object_id {object_id} not present in {self.origin or 'file'}")
        return self._groups[self._by_object_id[object_id]]

    def group_for_hdu(self, hdu_index: int) -> ObjectHduGroup:
        """Resolve the group whose COMBINED1D HDU is ``hdu_index``.

        The IRSA association table supplies this index directly, which avoids
        scanning the file when only one object is wanted.
        """
        name = self._extnames[hdu_index]
        m = _COMBINED_RE.match(name)
        if m is None:
            raise ValueError(f"HDU {hdu_index} is {name!r}, not a COMBINED1D_SIGNAL extension")
        return self.groups()[int(m.group(1))]

    # -- readers ----------------------------------------------------------
    def _common(self, hdu_index: int) -> dict[str, Any]:
        hdu = self._hdul[hdu_index]
        header = hdu.header
        data = hdu.data
        fs = flux_scale(header)
        return {
            "wavelength": np.asarray(data["WAVELENGTH"], dtype=np.float64),
            "flux": np.asarray(data["SIGNAL"], dtype=np.float64) * fs,
            "variance": np.asarray(data["VAR"], dtype=np.float64) * fs**2,
            "mask": np.asarray(data["MASK"], dtype=np.int64),
            "quality": np.asarray(data["QUALITY"], dtype=np.float64),
            "lsf_sigma": float(header["LSF_SIG"]),
            "bin_width": float(header.get("BINWIDTH", np.nan)),
            "mask_definition": self.mask_definition,
            "metadata": {
                "extname": header.get("EXTNAME", ""),
                "fscale": fs,
                "wmin": float(header.get("WMIN", np.nan)),
                "bincount": int(header.get("BINCOUNT", len(data))),
                "origin": self.origin,
                "tile_id": self.tile_id,
                "grism": self.grism_combination,
            },
            "_columns": list(data.columns.names),
            "_header": header,
            "_data": data,
        }

    def read_combined(self, group: ObjectHduGroup) -> CombinedSpectrum:
        c = self._common(group.combined_hdu)
        data = c.pop("_data")
        header = c.pop("_header")
        columns = c.pop("_columns")
        ndith = np.asarray(data["NDITH"], dtype=np.int32) if "NDITH" in columns else None
        return CombinedSpectrum(
            ndith=ndith,
            exposure_time=float(header.get("EXPTIME", np.nan)),
            extraction_profile=str(header.get("EXT_PROF", "")).strip(),
            **c,
        )

    def read_dithers(self, group: ObjectHduGroup) -> tuple[DitherSpectrum, ...]:
        out: list[DitherSpectrum] = []
        for pointing_id, hdu_index in group.dither_signal_hdus.items():
            c = self._common(hdu_index)
            c.pop("_data")
            header = c.pop("_header")
            c.pop("_columns")
            contam_hdu = group.dither_contam_hdus.get(pointing_id)
            contaminants: tuple[int, ...] = ()
            if contam_hdu is not None:
                cdata = self._hdul[contam_hdu].data
                if cdata is not None and len(cdata) > 0:
                    contaminants = tuple(int(v) for v in np.asarray(cdata["OBJ_ID"]))
            out.append(
                DitherSpectrum(
                    dither_id=int(header.get("DITH_ID", -1)),
                    pointing_id=int(header.get("PTGID", pointing_id)),
                    detector_id=int(header.get("DET_ID", -1)),
                    gwa_position=str(header.get("GWA_POS", "")).strip(),
                    gwa_tilt=float(header.get("GWA_TILT", np.nan)),
                    exposure_time=float(header.get("EXPTIME", np.nan)),
                    extraction_profile=str(header.get("EXT_PROF", "")).strip(),
                    contaminant_object_ids=contaminants,
                    **c,
                )
            )
        return tuple(sorted(out, key=lambda d: (d.dither_id, d.pointing_id)))

    def read_source_context(self, group: ObjectHduGroup) -> SourceContext:
        header = self._hdul[group.meta_hdu].header
        return SourceContext(
            object_id=int(header["OBJ_ID"]),
            ra=float(header["RA_OBJ"]),
            dec=float(header["DEC_OBJ"]),
            tile_id=self.tile_id,
            release="q1",
        )

    def read_observation(self, object_id: int, with_dithers: bool = True):
        """Return a :class:`~euclid_agn.spectra.types.SpectralObservation`."""
        from euclid_agn.spectra.types import SpectralObservation

        group = self.group_for_object(object_id)
        return SpectralObservation(
            source=self.read_source_context(group),
            combined=self.read_combined(group),
            dithers=self.read_dithers(group) if with_dithers else (),
        )

    def read_observation_at_hdu(self, hdu_index: int, with_dithers: bool = True):
        from euclid_agn.spectra.types import SpectralObservation

        group = self.group_for_hdu(hdu_index)
        return SpectralObservation(
            source=self.read_source_context(group),
            combined=self.read_combined(group),
            dithers=self.read_dithers(group) if with_dithers else (),
        )

    def close(self) -> None:
        self._hdul.close()

    def __enter__(self) -> SirCombinedSpectraFile:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


@contextmanager
def open_sir_file(
    path: str | Path,
    *,
    filesystem=None,
    anon: bool = True,
) -> Iterator[SirCombinedSpectraFile]:
    """Open a SIR combined-spectra file, locally or lazily from S3.

    Parameters
    ----------
    path : str or Path
        Either a local path, or an S3 key/URI such as
        ``s3://nasa-irsa-euclid-q1/q1/SIR/<tile>/EUC_SIR_W-COMBSPEC_*.fits``.
    filesystem : fsspec filesystem, optional
        Pre-built filesystem; if ``None`` and ``path`` looks like S3 an
        anonymous :class:`s3fs.S3FileSystem` is created.
    anon : bool
        Anonymous S3 access (IRSA's mirror is public).

    Notes
    -----
    Only the HDU headers plus the requested table bytes are transferred: a
    112 MB file is opened in a few seconds and a single object costs a few
    tens of kB.
    """
    text = str(path)
    is_s3 = text.startswith("s3://") or text.startswith("nasa-irsa-euclid")
    if filesystem is None and is_s3:
        import s3fs

        filesystem = s3fs.S3FileSystem(anon=anon)
    if filesystem is None:
        hdul = fits.open(text, lazy_load_hdus=True, memmap=False)
        try:
            yield SirCombinedSpectraFile(hdul, origin=text)
        finally:
            hdul.close()
    else:
        key = text.removeprefix("s3://")
        with filesystem.open(key, "rb") as fobj:
            hdul = fits.open(fobj, lazy_load_hdus=True, memmap=False)
            try:
                yield SirCombinedSpectraFile(hdul, origin=text)
            finally:
                hdul.close()
