"""IRSA Q1 backend: TAP catalogues plus lazy S3 access to SIR spectra.

Two facts about the IRSA Q1 service shape this implementation and were VERIFIED
against the live service on 2026-09-16:

1. ``euclid.objectid_spectrafile_association_q1`` lives in a different TAP
   datasource from ``euclid_q1_mer_catalogue``; server-side joins between them
   are rejected (``unknown datasource / table``).  Catalogue and association
   rows are therefore fetched separately and joined client-side, which also
   makes each query individually cacheable and loggable.

2. Most association rows carry a NULL ``path``: for tile 102160339 only
   18251 of 121547 rows have a spectrum.  ``path IS NOT NULL`` is the
   spectrum-availability filter.

The ``path`` column is an IRSA web-API URL of the form::

    api/spectrumdm/convert/euclid/q1/SIR/<tile>/EUC_SIR_W-COMBSPEC_<tile>_<ts>.fits?dataset_id=euclid_combspec&hdu=<n>

from which the S3 key ``nasa-irsa-euclid-q1/q1/SIR/<tile>/EUC_...fits`` is
derived by :func:`sir_s3_key`.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from pathlib import Path

import pandas as pd

from euclid_agn.archive import schema
from euclid_agn.archive.base import ArchiveBackend, SpectrumLocation
from euclid_agn.archive.tap import TapService, in_list_clause, quote_columns
from euclid_agn.io.sir import open_sir_file
from euclid_agn.spectra.types import CombinedSpectrum, DitherSpectrum, SourceContext

_PATH_RE = re.compile(r"(?P<rel>(?:SIR|MER|NIR|VIS)/[^?]+\.fits)")

#: Maximum number of identifiers put into a single ADQL ``IN`` clause.
ID_CHUNK = 500


def sir_s3_key(path: str, bucket_root: str = schema.IRSA_S3_ROOT) -> str:
    """Convert an association-table ``path`` into an S3 key.

    Accepts either the IRSA API URL form or a plain relative path.

    Examples
    --------
    >>> sir_s3_key("api/spectrumdm/convert/euclid/q1/SIR/1/EUC_SIR_W-COMBSPEC_1_x.fits?hdu=3")
    'nasa-irsa-euclid-q1/q1/SIR/1/EUC_SIR_W-COMBSPEC_1_x.fits'
    """
    m = _PATH_RE.search(path)
    if m is None:
        raise ValueError(f"cannot derive an S3 key from association path {path!r}")
    return f"{bucket_root}/{m.group('rel')}"


class IrsaQ1Backend(ArchiveBackend):
    """Archive backend for Euclid Q1 served by IRSA."""

    release = "q1"

    def __init__(
        self,
        *,
        tap: TapService | None = None,
        filesystem=None,
        anon: bool = True,
        timeout: float = 900.0,
    ) -> None:
        self.tap = tap or TapService(schema.IRSA_TAP_SYNC, timeout=timeout)
        self._filesystem = filesystem
        self.anon = anon

    # -- filesystem -------------------------------------------------------
    @property
    def filesystem(self):
        if self._filesystem is None:
            import s3fs

            self._filesystem = s3fs.S3FileSystem(anon=self.anon)
        return self._filesystem

    # -- catalogue queries ------------------------------------------------
    def list_tiles(self, field: str) -> list[int]:
        """Tile IDs inside a named field cone.

        Resolved through the CAOM tile/plane tables rather than the MER
        catalogue: ``SELECT DISTINCT tileid FROM euclid_q1_mer_catalogue WHERE
        ra BETWEEN ... AND dec BETWEEN ...`` was measured to fail after five
        minutes on the live service, while this query returns in ~15 s.
        """
        fp = schema.FIELDS[field]
        adql = (
            f"SELECT DISTINCT t.tileid FROM {schema.CAOM_TILE_ASSOCIATION} AS t "
            f"JOIN {schema.CAOM_PLANE} AS p ON p.obsid = t.obsid "
            f"WHERE {fp.cone_adql('p.pt')}"
        )
        return sorted(int(v) for v in self.tap.query(adql)["tileid"])

    def query_spectrum_locations(
        self,
        *,
        field: str | None = None,
        tile_id: int | Sequence[int] | None = None,
        object_ids: Sequence[int] | None = None,
        limit: int | None = None,
    ) -> list[SpectrumLocation]:
        df = self.query_association(
            field=field, tile_id=tile_id, object_ids=object_ids, limit=limit
        )
        return [
            SpectrumLocation(
                object_id=int(r.objectid),
                tile_id=int(r.tileid),
                path=sir_s3_key(str(r.path)),
                hdu=int(r.hdu),
                bandpass=str(getattr(r, "bandpass_name", "")),
                release=self.release,
            )
            for r in df.itertuples()
        ]

    def query_association(
        self,
        *,
        field: str | None = None,
        tile_id: int | Sequence[int] | None = None,
        object_ids: Sequence[int] | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """Rows of the objectid/spectrum-file association table that have a spectrum."""
        cols = "objectid, tileid, path, hdu, bandpass_name"
        where = ["path IS NOT NULL"]
        if object_ids is not None:
            frames = [
                self.tap.query(
                    self._select(
                        cols,
                        schema.SPECTRA_ASSOCIATION,
                        where + [in_list_clause("objectid", chunk)],
                        limit=None,
                    )
                )
                for chunk in _chunks(object_ids, ID_CHUNK)
            ]
            df = pd.concat(frames, ignore_index=True) if frames else _empty_association()
        else:
            if tile_id is not None:
                where.append(self._tile_clause("tileid", tile_id))
            elif field is not None:
                where.append(self._tile_clause("tileid", self.list_tiles(field)))
            df = self.tap.query(self._select(cols, schema.SPECTRA_ASSOCIATION, where, limit))
        if limit is not None:
            df = df.head(limit)
        return df.reset_index(drop=True)

    def query_sources(
        self,
        *,
        field: str | None = None,
        tile_id: int | Sequence[int] | None = None,
        object_ids: Sequence[int] | None = None,
        limit: int | None = None,
        require_spectrum: bool = True,
        with_morphology: bool = True,
        with_phz: bool = True,
        with_spe: bool = True,
    ) -> pd.DataFrame:
        """One row per source: MER context joined to the spectrum address.

        Every catalogue column here is *context*.  None of it may be used as an
        AGN-selection criterion; it exists so the selection function can be
        reported against host and data properties.
        """
        assoc = (
            self.query_association(
                field=field, tile_id=tile_id, object_ids=object_ids, limit=limit
            )
            if require_spectrum
            else _empty_association()
        )
        ids = (
            list(assoc["objectid"])
            if require_spectrum
            else (list(object_ids) if object_ids is not None else None)
        )
        if require_spectrum and not ids:
            return pd.DataFrame(columns=["object_id"])

        mer = self._query_by_ids(
            schema.MER_CATALOGUE, schema.MER_MANIFEST_COLUMNS, "object_id", ids,
            field=field, tile_id=tile_id, limit=limit,
        )
        out = mer.rename(columns={"tileid": "tile_id"})
        if require_spectrum:
            loc = assoc.rename(columns={"objectid": "object_id", "tileid": "tile_id"})
            loc["sir_s3_key"] = [sir_s3_key(p) for p in loc["path"]]
            loc = loc.rename(columns={"path": "sir_api_path", "hdu": "sir_hdu"})
            out = out.merge(loc.drop(columns=["tile_id"]), on="object_id", how="inner")

        ids = list(out["object_id"])
        if with_morphology:
            out = self._left_join(
                out, schema.MER_MORPHOLOGY, schema.MORPHOLOGY_MANIFEST_COLUMNS, ids
            )
        if with_phz:
            out = self._left_join(out, schema.PHZ_PHOTO_Z, schema.PHZ_MANIFEST_COLUMNS, ids)
        if with_spe:
            out = self._left_join(
                out, schema.SPE_CLASSIFICATION, schema.SPE_CLASSIFICATION_COLUMNS, ids
            )
            out = self._left_join(out, schema.SPE_QUALITY, schema.SPE_QUALITY_COLUMNS, ids)
            gal = self._query_by_ids(
                schema.SPE_GALAXY_CANDIDATES,
                ("object_id", "spe_rank", "spe_z", "spe_z_err", "spe_z_prob", "spe_cont_snr"),
                "object_id",
                ids,
                extra_where=["spe_rank = 0"],
            )
            gal = gal.drop(columns=["spe_rank"]).rename(
                columns={
                    "spe_z": "spe_gal_z",
                    "spe_z_err": "spe_gal_z_err",
                    "spe_z_prob": "spe_gal_z_prob",
                }
            )
            out = out.merge(gal, on="object_id", how="left")
            qso = self._query_by_ids(
                schema.SPE_QSO_CANDIDATES,
                ("object_id", "spe_rank", "spe_z", "spe_z_err", "spe_z_prob"),
                "object_id",
                ids,
                extra_where=["spe_rank = 0"],
            )
            qso = qso.drop(columns=["spe_rank"]).rename(
                columns={
                    "spe_z": "spe_qso_z",
                    "spe_z_err": "spe_qso_z_err",
                    "spe_z_prob": "spe_qso_z_prob",
                }
            )
            out = out.merge(qso, on="object_id", how="left")
        out["release"] = self.release
        return out.reset_index(drop=True)

    # -- spectra ----------------------------------------------------------
    def open_observation(self, location: SpectrumLocation, with_dithers: bool = True):
        """Read one object's spectra straight from the S3 mirror."""
        with open_sir_file(location.path, filesystem=self.filesystem, anon=self.anon) as sir:
            return sir.read_observation_at_hdu(location.hdu, with_dithers=with_dithers)

    def open_combined_spectrum(self, location: SpectrumLocation) -> CombinedSpectrum:
        return self.open_observation(location, with_dithers=False).combined

    def open_dither_spectra(self, location: SpectrumLocation) -> tuple[DitherSpectrum, ...]:
        return self.open_observation(location, with_dithers=True).dithers

    def get_source_context(self, object_id: int) -> SourceContext:
        df = self.query_sources(object_ids=[object_id])
        if df.empty:
            raise KeyError(f"object_id {object_id} has no Q1 spectrum association")
        row = df.iloc[0]
        return SourceContext(
            object_id=int(row["object_id"]),
            ra=float(row["ra"]),
            dec=float(row["dec"]),
            tile_id=int(row["tile_id"]),
            release=self.release,
            photometry={
                k: float(row[k]) for k in df.columns if k.startswith("flux_") and pd.notna(row[k])
            },
            morphology={
                k: float(row[k])
                for k in schema.MORPHOLOGY_MANIFEST_COLUMNS
                if k in df.columns and pd.notna(row[k])
            },
            photometric_redshift={
                k: float(row[k])
                for k in ("phz_median", "phz_mode_1", "phz_mode_2")
                if k in df.columns and pd.notna(row[k])
            },
        )

    def download_file(self, location: SpectrumLocation, destination: str | Path) -> Path:
        """Copy a whole SIR file locally (for regression fixtures; not for bulk runs)."""
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.filesystem.get(location.path, str(destination))
        return destination

    # -- query helpers ----------------------------------------------------
    @staticmethod
    def _tile_clause(column: str, tile_id) -> str:
        if isinstance(tile_id, int):
            return f"{column} = {tile_id}"
        return in_list_clause(column, tile_id)

    @staticmethod
    def _select(columns, table: str, where: Sequence[str], limit: int | None) -> str:
        top = f"TOP {int(limit)} " if limit else ""
        clause = " AND ".join(w for w in where if w)
        tail = f" WHERE {clause}" if clause else ""
        return f"SELECT {top}{quote_columns(columns)} FROM {table}{tail}"

    def _query_by_ids(
        self,
        table: str,
        columns: Sequence[str],
        id_column: str,
        ids: Sequence[int] | None,
        *,
        field: str | None = None,
        tile_id=None,
        limit: int | None = None,
        extra_where: Sequence[str] = (),
    ) -> pd.DataFrame:
        cols = list(dict.fromkeys([id_column, *columns]))
        if ids is not None:
            frames = [
                self.tap.query(
                    self._select(
                        cols, table, [in_list_clause(id_column, chunk), *extra_where], None
                    )
                )
                for chunk in _chunks(ids, ID_CHUNK)
            ]
            if not frames:
                return pd.DataFrame(columns=list(dict.fromkeys([id_column, *columns])))
            return pd.concat(frames, ignore_index=True)
        where = list(extra_where)
        if tile_id is not None:
            where.append(self._tile_clause("tileid", tile_id))
        elif field is not None:
            where.append(self._tile_clause("tileid", self.list_tiles(field)))
        return self.tap.query(self._select(cols, table, where, limit))

    def _left_join(
        self, frame: pd.DataFrame, table: str, columns: Sequence[str], ids: Sequence[int]
    ) -> pd.DataFrame:
        extra = self._query_by_ids(table, columns, "object_id", ids)
        extra = extra.drop(columns=[c for c in ("cntr",) if c in extra.columns])
        overlap = [c for c in extra.columns if c in frame.columns and c != "object_id"]
        extra = extra.drop(columns=overlap)
        return frame.merge(extra, on="object_id", how="left")


def _chunks(values: Iterable[int], size: int) -> list[list[int]]:
    values = list(values)
    return [values[i : i + size] for i in range(0, len(values), size)]


def _empty_association() -> pd.DataFrame:
    return pd.DataFrame(columns=["objectid", "tileid", "path", "hdu", "bandpass_name"])
