"""Point sources with Euclid spectra around nearby galaxies: globular-cluster / UCD test set.

Hosts are nearby (D < 40 Mpc) galaxies from HyperLEDA inside the Q1 field cones.
For each host the Q1 tiles covering it are found through the CAOM cone lookup
(fast), then MER is filtered by ``tileid`` and an RA/Dec box (indexed columns;
a ``CONTAINS`` cone on the 30 M-row table takes > 10 min).  Candidates:
``has_spectrum``, point-like, H_AB < h_max, within ``radius`` of the host.
The host's heliocentric velocity is the ground truth for every member.

Usage::

    python -m euclid_agn.validation.gc_hosts --out outputs/gc_candidates.parquet
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from euclid_agn.archive import schema
from euclid_agn.archive.irsa import sir_s3_key
from euclid_agn.archive.tap import TapService

log = logging.getLogger(__name__)

#: name, field, ra, dec, D/Mpc, type, v_helio (HyperLEDA, WSDB leda.main, 2026-09-17)
HOSTS = [
    ("NGC1527", "EDF-S", 62.1005, -47.8970, 16.8, "E-S0", 1175.5), ("NGC1494", "EDF-S", 59.4277, -48.9115, 16.1, "Scd", 1126.6),
    ("NGC1483", "EDF-S", 58.1983, -47.4775, 16.3, "SBbc", 1144.4), ("IC2035", "EDF-S", 62.2580, -45.5176, 21.5, "E-S0", 1503.9),
    ("NGC1493", "EDF-S", 59.3644, -46.2109, 15.0, "SBc", 1053.3), ("IC2000", "EDF-S", 57.2823, -48.8582, 14.0, "SBc", 980.4),
    ("NGC1556", "EDF-S", 64.4368, -50.1644, 14.1, "Sb", 985.3), ("NGC1433", "EDF-S", 55.5064, -47.2221, 15.4, "SBa", 1075.4),
    ("NGC1398", "EDF-F", 54.7172, -26.3379, 20.0, "SBab", 1399.8), ("NGC1340", "EDF-F", 52.0817, -31.0681, 16.9, "E", 1183.1),
    ("NGC1425", "EDF-F", 55.5478, -29.8934, 21.6, "Sb", 1510.7), ("NGC1412", "EDF-F", 55.1224, -26.8623, 25.6, "S0", 1790.1),
    ("ESO418-008", "EDF-F", 52.8776, -30.2132, 17.1, "SABd", 1194.3), ("NGC1366", "EDF-F", 53.4737, -31.1941, 18.7, "S0", 1307.5),
    ("NGC1339", "EDF-F", 52.0274, -32.2861, 19.9, "E", 1391.5), ("NGC1406", "EDF-F", 54.8471, -31.3214, 15.4, "SBbc", 1075.4),
    ("NGC2110", "LDN1641", 88.0475, -7.4561, 33.0, "E-S0 Sy2", 2311.7), ("NGC6667", "EDF-N", 277.6659, 67.9870, 36.9, "SABa", 2586.3),
    ("NGC6395", "EDF-N", 261.6303, 71.0963, 16.6, "Sc", 1163.1), ("NGC6503", "EDF-N", 267.3605, 70.1443, 0.5, "Sc", 35.9),
]


def tiles_covering(tap: TapService, ra: float, dec: float, radius_deg: float = 0.6) -> pd.DataFrame:
    """Q1 tiles whose CAOM centre lies within ``radius_deg`` of (ra, dec).

    ``CONTAINS(p.pt, CIRCLE(...))`` - argument order matters: the reverse form
    and ``INTERSECTS`` are both rejected by the service.  ~6 min per call, so
    results are cached in ``outputs/gc_host_tiles.parquet``.
    """
    adql = (f"SELECT DISTINCT t.tileid, COORD1(p.pt) AS tile_ra, COORD2(p.pt) AS tile_dec "
            f"FROM {schema.CAOM_TILE_ASSOCIATION} AS t JOIN {schema.CAOM_PLANE} AS p ON p.obsid = t.obsid "
            f"WHERE CONTAINS(p.pt, CIRCLE('ICRS', {ra}, {dec}, {radius_deg})) = 1")
    return tap.query(adql)


def host_tiles(hosts=None, cache=Path("outputs/gc_host_tiles.parquet"), radius_deg: float = 0.6) -> pd.DataFrame:
    """Tile list per host, accumulated across runs (one slow CAOM call per new host)."""
    done = pd.read_parquet(cache) if cache.exists() else pd.DataFrame(columns=["host", "tileid", "tile_ra", "tile_dec"])
    tap = TapService(schema.IRSA_TAP_SYNC)
    frames = [done]
    for name, field, ra, dec, d, typ, v in (hosts or HOSTS):
        if name in set(done.host):
            continue
        try:
            t = tiles_covering(tap, ra, dec, radius_deg)
        except Exception as exc:  # noqa: BLE001
            log.warning("%s: CAOM lookup failed (%s)", name, str(exc)[:80])
            continue
        if len(t):
            t = t.assign(host=name)
            frames.append(t)
            log.info("%s: %d tiles", name, len(t))
        pd.concat(frames, ignore_index=True).to_parquet(cache, index=False)
    return pd.concat(frames, ignore_index=True)


def sir_files_for_tiles(tiles, backend=None) -> pd.DataFrame:
    """SIR file paths for the given tiles (association table; the tested query path)."""
    from euclid_agn.pipeline.ingest import make_backend

    backend = backend or make_backend()
    frames = []
    for tile in sorted({int(t) for t in tiles}):
        try:
            a = backend.query_association(tile_id=tile)
        except Exception as exc:  # noqa: BLE001
            log.warning("tile %s: association query failed (%s)", tile, str(exc)[:80])
            continue
        if len(a):
            frames.append(a.assign(tileid=tile))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def objects_near(sir_path, ra: float, dec: float, radius_arcmin: float) -> pd.DataFrame:
    """Object IDs, positions and spectrum quality within a radius, read from a SIR file's META HDUs."""
    from euclid_agn.io.sir import open_sir_file

    cosd = np.cos(np.deg2rad(dec))
    rows = []
    with open_sir_file(sir_path) as f:
        for oid in f.object_ids():
            group = f.group_for_object(oid)
            ctx = f.read_source_context(group)
            sep = 60.0 * np.hypot((ctx.ra - ra) * cosd, ctx.dec - dec)
            if sep > radius_arcmin:
                continue
            obs = f.read_observation(oid, with_dithers=False)
            spectrum = obs.combined
            ok = spectrum.usable()
            snr = float(np.nanmedian(spectrum.flux[ok] / np.sqrt(spectrum.variance[ok]))) if ok.any() else np.nan
            rows.append({"object_id": int(oid), "ra": float(ctx.ra), "dec": float(ctx.dec), "sep_arcmin": float(sep),
                         "snr": snr, "usable_fraction": float(ok.mean()), "lsf_sigma": float(spectrum.lsf_sigma),
                         "file": str(sir_path)})
    return pd.DataFrame(rows)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--hosts", nargs="*", default=None, help="host names (default: all)")
    parser.add_argument("--radius-arcmin", type=float, default=10.0)
    parser.add_argument("--min-snr", type=float, default=3.0)
    parser.add_argument("--out", type=Path, default=Path("outputs/gc_candidates.parquet"))
    parser.add_argument("--tiles-only", action="store_true", help="only resolve host -> tiles and cache them")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    hosts = [h for h in HOSTS if args.hosts is None or h[0] in args.hosts]
    tiles = host_tiles(hosts)
    print(tiles.groupby("host").size().to_string() if len(tiles) else "no tiles resolved")
    if args.tiles_only:
        return
    from euclid_agn.io.cache import ArchiveCache
    from euclid_agn.pipeline.ingest import make_backend

    backend = make_backend()
    cache = ArchiveCache(Path("~/data/euclid").expanduser())
    frames = []
    for name, field, ra, dec, d, typ, v in hosts:
        mine = tiles[tiles.host == name]
        if not len(mine):
            continue
        # the tile whose centre is nearest the host is the one that contains it
        cosd = np.cos(np.deg2rad(dec))
        mine = mine.assign(sep=np.hypot((mine.tile_ra - ra) * cosd, mine.tile_dec - dec)).nsmallest(4, "sep")
        files = sir_files_for_tiles(mine.tileid, backend)
        if not len(files):
            log.warning("%s: no SIR files for tiles %s", name, list(mine.tileid))
            continue
        for path in sorted({p for p in files.path}):
            key = sir_s3_key(path)
            local = cache.local_path(key)
            if not local.is_file():
                log.info("%s: downloading %s", name, key)
                try:
                    cache.fetch(key, backend.filesystem)
                except Exception as exc:  # noqa: BLE001
                    log.warning("%s: download failed (%s)", name, str(exc)[:80])
                    continue
            near = objects_near(local, ra, dec, args.radius_arcmin)
            if len(near):
                frames.append(near.assign(host=name, field=field, host_d_mpc=d, host_type=typ, host_v=v))
                log.info("%s: %d objects within %.0f' in %s (S/N>%.0f: %d)", name, len(near), args.radius_arcmin,
                         local.name, args.min_snr, int((near.snr > args.min_snr).sum()))
    if not frames:
        print("no candidates found")
        return
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(args.out, index=False)
    good = out[out.snr > args.min_snr]
    print(f"\n{len(out)} objects within {args.radius_arcmin:.0f}' of the hosts; {len(good)} with S/N > {args.min_snr}")
    print(out.groupby("host").agg(n=("object_id", "size"), n_snr=("snr", lambda s: int((s > args.min_snr).sum())),
                                  snr_max=("snr", "max")).to_string())


if __name__ == "__main__":
    main()
