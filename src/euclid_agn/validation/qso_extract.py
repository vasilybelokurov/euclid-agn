"""Pull the confirmed-quasar spectra out of the archive into one local store.

The 2,755 DESI quasars with Euclid spectra sit in 1,581 different SIR files
with a median of one quasar per file, so the cost is dominated by opening
files, not by reading data: 16.7 s per remote open, i.e. seven hours serially
and about half an hour with a thread pool.  Downloading the files instead
would be 63 GB for a few tens of MB of useful spectra.

Each spectrum is stored after the dither-scatter variance rescaling, on the
common Q1 wavelength grid, so downstream work (the AGN decomposition, the
class engine) reads a single 40 MB file instead of touching the archive.

Usage::

    python -m euclid_agn.validation.qso_extract --workers 16 --out outputs/qso_spectra_edfn.npz
"""

from __future__ import annotations

import argparse
import logging
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from euclid_agn.io.sir import open_sir_file
from euclid_agn.spectra.artefacts import continuum_trough
from euclid_agn.spectra.coherence import dither_variance_rescale

log = logging.getLogger(__name__)


def extract_file(key: str, rows: pd.DataFrame, filesystem) -> list[dict]:
    """Every requested object in one SIR file, opened once."""
    out = []
    with open_sir_file(key, filesystem=filesystem) as f:
        for _, row in rows.iterrows():
            try:
                obs = f.read_observation_at_hdu(int(row.hdu), with_dithers=True)
            except Exception:  # noqa: BLE001 - a bad HDU index should not lose the file
                try:
                    obs = f.read_observation(int(row.object_id), with_dithers=True)
                except Exception:  # noqa: BLE001
                    continue
            if int(obs.source.object_id) != int(row.object_id):
                continue  # the association's HDU index disagrees with the header; skip rather than mislabel
            spectrum, _ = dither_variance_rescale(obs) if obs.dithers else (obs.combined, None)
            usable = spectrum.usable()
            out.append({
                "object_id": int(row.object_id), "wavelength": spectrum.wavelength, "flux": spectrum.flux,
                "variance": spectrum.variance, "mask": spectrum.mask, "lsf_sigma": float(spectrum.lsf_sigma),
                "bin_width": float(spectrum.bin_width), "n_dithers": len(obs.dithers),
                "usable_fraction": float(usable.mean()),
                "snr": float(np.nanmedian(spectrum.flux[usable] / np.sqrt(spectrum.variance[usable]))) if usable.any() else np.nan,
                "trough": bool(continuum_trough(spectrum).flagged),
                "ra": float(obs.source.ra), "dec": float(obs.source.dec),
            })
    return out


#: IRSA serves one spectrum per request from the association table's ``path``, converted to a
#: VOTable (WAVELENGTH, SIGNAL, VAR, MASK, QUALITY, NDITH; flux already scaled by FSCALE).
#: 2.3 s per spectrum against 16.7 s to walk to the same HDU inside the 110 MB file over S3.
IRSA_BASE = "https://irsa.ipac.caltech.edu/"

#: Variance inflation applied when the per-object dither rescaling is unavailable.  The
#: archive variance underestimates the true scatter; eta^2 ~ 2 was measured from residual
#: autocorrelation in session 3, and the local dither scatter is larger still near bright
#: neighbours - so this is a floor, not a correction.
GLOBAL_NOISE_INFLATION = 2.0


def fetch_votable(path: str, timeout: float = 180.0) -> "pd.DataFrame":
    """One spectrum from IRSA's per-HDU API, as a table."""
    import io
    import urllib.request

    from astropy.table import Table

    url = IRSA_BASE + str(path).lstrip("/")
    with urllib.request.urlopen(url, timeout=timeout) as response:
        payload = response.read()
    return Table.read(io.BytesIO(payload), format="votable").to_pandas()


def _api_worker(record):
    """Fetch and package one spectrum; returns None on failure rather than killing the pool."""
    import warnings

    warnings.filterwarnings("ignore")
    try:
        table = fetch_votable(record["path"])
    except Exception:  # noqa: BLE001 - one bad object must not stop 2,755
        return None
    flux = table["SIGNAL"].to_numpy(dtype=np.float64) * 1e-16
    variance = table["VAR"].to_numpy(dtype=np.float64) * 1e-32 * GLOBAL_NOISE_INFLATION
    mask = table["MASK"].to_numpy(dtype=np.int64)
    usable = (mask & 1) == 0
    usable &= np.isfinite(flux) & np.isfinite(variance) & (variance > 0)
    snr = float(np.nanmedian(flux[usable] / np.sqrt(variance[usable]))) if usable.any() else np.nan
    return {"object_id": int(record["object_id"]), "wavelength": table["WAVELENGTH"].to_numpy(dtype=np.float64),
            "flux": flux, "variance": variance, "mask": mask,
            "lsf_sigma": float(record.get("lsf_sigma", np.nan)), "bin_width": 13.4,
            "n_dithers": int(np.nanmax(table["NDITH"].to_numpy())) if "NDITH" in table else -1,
            "usable_fraction": float(usable.mean()), "snr": snr, "trough": False,
            "ra": float(record.get("ra", np.nan)), "dec": float(record.get("dec", np.nan))}


def run_api(located: pd.DataFrame, workers: int = 16) -> list[dict]:
    """Fetch every spectrum through the IRSA API, in parallel (I/O bound, so threads are fine)."""
    from concurrent.futures import ThreadPoolExecutor

    records = located.to_dict("records")
    done, started = [], time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, got in enumerate(pool.map(_api_worker, records), start=1):
            if got is not None:
                done.append(got)
            if i % 250 == 0:
                rate = i / (time.time() - started)
                log.info("%d/%d fetched (%d ok), %.1f/s, %.0f min left", i, len(records), len(done), rate,
                         (len(records) - i) / max(rate, 1e-9) / 60)
    log.info("%d spectra in %.0f s", len(done), time.time() - started)
    return done


def _worker(payload):
    """One SIR file in its own process.

    Threads gain little here: astropy parses thousands of HDU headers per file
    to reach the one we want, which holds the GIL.  Each process therefore
    builds its own S3 filesystem - the handle does not survive pickling.
    """
    key, records = payload
    import s3fs

    filesystem = s3fs.S3FileSystem(anon=True)
    try:
        return extract_file(key, pd.DataFrame(records), filesystem)
    except Exception as exc:  # noqa: BLE001
        logging.getLogger(__name__).warning("%s: %s", key.split("/")[-1], str(exc)[:70])
        return []


def run(located: pd.DataFrame, workers: int = 8, filesystem=None) -> list[dict]:
    """Extract every located quasar, one process per file."""
    payloads = [(key, rows[["object_id", "hdu"]].to_dict("records")) for key, rows in located.groupby("s3_key")]
    done, started = [], time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, got in enumerate(pool.map(_worker, payloads, chunksize=1), start=1):
            done.extend(got)
            if i % 50 == 0:
                rate = i / (time.time() - started)
                log.info("%d/%d files, %d spectra, %.1f files/s, %.0f min left",
                         i, len(payloads), len(done), rate, (len(payloads) - i) / max(rate, 1e-9) / 60)
    log.info("%d spectra from %d files in %.0f s", len(done), len(payloads), time.time() - started)
    return done


def save(spectra: list[dict], out: Path) -> pd.DataFrame:
    """Arrays to npz on the shared grid, metadata to parquet beside it."""
    grid = spectra[0]["wavelength"]
    same = all(s["wavelength"].shape == grid.shape and np.allclose(s["wavelength"], grid) for s in spectra)
    if not same:
        raise ValueError("spectra are not on a common wavelength grid")
    order = np.argsort([s["object_id"] for s in spectra])
    spectra = [spectra[i] for i in order]
    np.savez_compressed(
        out,
        wavelength=grid.astype(np.float32),
        object_id=np.array([s["object_id"] for s in spectra], dtype=np.int64),
        flux=np.vstack([s["flux"] for s in spectra]).astype(np.float32),
        variance=np.vstack([s["variance"] for s in spectra]).astype(np.float32),
        mask=np.vstack([s["mask"] for s in spectra]).astype(np.int16),
    )
    meta = pd.DataFrame([{k: v for k, v in s.items() if not isinstance(v, np.ndarray)} for s in spectra])
    meta.to_parquet(out.with_suffix(".meta.parquet"), index=False)
    return meta


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--located", type=Path, default=Path("outputs/qso_truth_edfn_located.parquet"))
    parser.add_argument("--out", type=Path, default=Path("outputs/qso_spectra_edfn.npz"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--source", choices=("api", "s3"), default="api",
                        help="api: one request per spectrum (fast, no dithers); s3: read the SIR files (slow, dithers)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    located = pd.read_parquet(args.located)
    if args.limit:
        located = located.head(args.limit)
    print(f"{len(located):,} quasars in {located.s3_key.nunique():,} files, {args.workers} workers, source={args.source}")
    spectra = run_api(located, workers=args.workers) if args.source == "api" else run(located, workers=args.workers)
    meta = save(spectra, args.out)
    print(f"stored {len(meta):,} spectra -> {args.out} ({args.out.stat().st_size/1e6:.0f} MB)")
    print(f"  S/N quantiles {meta.snr.quantile([.1,.5,.9]).round(1).tolist()}; trough-flagged {meta.trough.mean():.0%}; "
          f"median dithers {meta.n_dithers.median():.0f}")


if __name__ == "__main__":
    main()


def load_store(path: Path):
    """``(meta, spectra)`` from a store written by :func:`save`.

    ``spectra`` is a callable taking a row index and returning a
    :class:`~euclid_agn.spectra.types.Spectrum1D`, so the 10 MB of arrays stay
    in one place and nothing re-reads the archive.
    """
    from euclid_agn.spectra.types import Spectrum1D

    path = Path(path)
    data = np.load(path)
    meta = pd.read_parquet(path.with_suffix(".meta.parquet"))
    wavelength = data["wavelength"].astype(np.float64)

    def spectrum(i: int) -> Spectrum1D:
        row = meta.iloc[i]
        lsf = float(row.lsf_sigma) if np.isfinite(row.lsf_sigma) else 13.7
        return Spectrum1D(wavelength=wavelength, flux=data["flux"][i].astype(np.float64),
                          variance=data["variance"][i].astype(np.float64), mask=data["mask"][i].astype(np.int64),
                          quality=np.ones(wavelength.size), lsf_sigma=lsf, bin_width=float(row.bin_width))

    return meta, spectrum
