"""Deterministic Euclid-like spectral simulator.

Two jobs:

1. generate synthetic :class:`~euclid_agn.spectra.types.SpectralObservation`
   objects with a known truth, for unit tests and injection/recovery;
2. write them out in the real ``DpdSirCombinedSpectra`` FITS layout, so the IO
   layer is tested against the same structure it will meet in the archive
   without any network access.

Everything is seeded: the same seed gives the same arrays on any machine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from astropy.io import fits

from euclid_agn.constants import (
    SIR_BINCOUNT,
    SIR_BINWIDTH_ANGSTROM,
    SIR_WMIN_ANGSTROM,
)
from euclid_agn.models.line_catalog import BY_NAME
from euclid_agn.spectra.lsf import (
    effective_sigma,
    gaussian_pixel_integral,
    sigma_kms_to_angstrom,
)
from euclid_agn.spectra.masking import MASK_BITS

DEFAULT_FSCALE = 1.0e-16


def sir_wavelength_grid(
    n: int = SIR_BINCOUNT,
    wmin: float = SIR_WMIN_ANGSTROM,
    bin_width: float = SIR_BINWIDTH_ANGSTROM,
) -> np.ndarray:
    """The real Q1 red-grism grid: 531 bins of 13.4 A from 11900 A."""
    return wmin + bin_width * np.arange(n, dtype=np.float64)


@dataclass(frozen=True)
class LineTruth:
    """One emission line to inject."""

    name: str
    flux: float  # erg/s/cm2, integrated
    sigma_kms: float  # intrinsic velocity dispersion
    velocity_kms: float = 0.0  # offset from systemic
    broad: bool = False


@dataclass(frozen=True)
class SpectrumTruth:
    """Full generative truth for one simulated spectrum."""

    z: float = 1.3
    continuum_flux: float = 1.0e-17  # erg/s/cm2/A at the pivot
    continuum_slope: float = 0.0  # per 1000 A
    pivot: float = 15000.0
    lines: tuple[LineTruth, ...] = ()
    lsf_sigma: float = 13.7  # Angstrom; Q1 median for compact sources
    noise_flux: float = 2.0e-18  # per-pixel sigma of the flux density
    seed: int = 0
    object_id: int = 1
    ra: float = 270.0
    dec: float = 66.0
    n_dithers: int = 4
    masked_fraction: float = 0.0
    metadata: dict = field(default_factory=dict)


def continuum_model(wavelength: np.ndarray, truth: SpectrumTruth) -> np.ndarray:
    """Linear continuum in flux density."""
    return truth.continuum_flux * (
        1.0 + truth.continuum_slope * (wavelength - truth.pivot) / 1000.0
    )


def line_model(wavelength: np.ndarray, truth: SpectrumTruth, bin_width: float) -> np.ndarray:
    """Sum of LSF-convolved, pixel-integrated emission lines."""
    out = np.zeros_like(wavelength)
    for line in truth.lines:
        rest = BY_NAME[line.name].rest
        centre = rest * (1.0 + truth.z) * (1.0 + line.velocity_kms / 299792.458)
        intrinsic = sigma_kms_to_angstrom(line.sigma_kms, centre)
        sigma = effective_sigma(intrinsic, truth.lsf_sigma)
        out += line.flux * gaussian_pixel_integral(wavelength, centre, sigma, bin_width)
    return out


def noiseless_spectrum(truth: SpectrumTruth, wavelength: np.ndarray | None = None):
    """Model flux density without noise, and the wavelength grid."""
    wavelength = sir_wavelength_grid() if wavelength is None else np.asarray(wavelength)
    bin_width = float(np.median(np.diff(wavelength))) if wavelength.size > 1 else 1.0
    flux = continuum_model(wavelength, truth) + line_model(wavelength, truth, bin_width)
    return wavelength, flux


def simulate_arrays(
    truth: SpectrumTruth,
    wavelength: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
    exposure_scale: float = 1.0,
):
    """Noisy realisation: wavelength, flux, variance, mask, quality.

    The noise is heteroscedastic: the per-pixel sigma grows as the square root
    of the model flux above the floor, mimicking photon noise on top of a
    read/background floor.
    """
    rng = rng or np.random.default_rng(truth.seed)
    wavelength, model = noiseless_spectrum(truth, wavelength)
    floor = truth.noise_flux / np.sqrt(exposure_scale)
    excess = np.clip(model - np.median(model), 0.0, None)
    sigma = np.sqrt(floor**2 + floor * excess)
    flux = model + rng.normal(0.0, sigma)
    variance = sigma**2
    mask = np.zeros(wavelength.size, dtype=np.int64)
    quality = np.ones(wavelength.size, dtype=np.float64)
    if truth.masked_fraction > 0:
        n_bad = int(round(truth.masked_fraction * wavelength.size))
        bad = rng.choice(wavelength.size, size=n_bad, replace=False)
        mask[bad] |= MASK_BITS["NOT_USE"]
        quality[bad] = 0.0
    return wavelength, flux, variance, mask, quality


def simulate_observation(truth: SpectrumTruth):
    """A full :class:`SpectralObservation` with combined and dither spectra."""
    from euclid_agn.spectra.types import (
        CombinedSpectrum,
        DitherSpectrum,
        SourceContext,
        SpectralObservation,
    )

    rng = np.random.default_rng(truth.seed)
    bin_width = SIR_BINWIDTH_ANGSTROM
    gwa = ("RGS000", "RGS180", "RGS000", "RGS180")
    dithers = []
    for i in range(truth.n_dithers):
        w, f, v, m, q = simulate_arrays(truth, rng=rng, exposure_scale=1.0)
        dithers.append(
            DitherSpectrum(
                wavelength=w,
                flux=f,
                variance=v,
                mask=m,
                quality=q,
                lsf_sigma=truth.lsf_sigma,
                bin_width=bin_width,
                dither_id=i,
                pointing_id=11889 + i,
                detector_id=32,
                gwa_position=gwa[i % len(gwa)],
                gwa_tilt=float(i),
                exposure_time=549.6422,
                extraction_profile="NONE",
                contaminant_object_ids=tuple(9_000_000_000 + i * 10 + j for j in range(i % 3)),
                metadata={"simulated": True},
            )
        )
    # Co-addition improves the noise by sqrt(N) at fixed exposure per dither.
    w, f, v, m, q = simulate_arrays(truth, rng=rng, exposure_scale=max(truth.n_dithers, 1))
    combined = CombinedSpectrum(
        wavelength=w,
        flux=f,
        variance=v,
        mask=m,
        quality=q,
        lsf_sigma=truth.lsf_sigma,
        bin_width=bin_width,
        ndith=np.full(w.size, truth.n_dithers, dtype=np.int32),
        exposure_time=549.6422 * truth.n_dithers,
        extraction_profile="NONE",
        metadata={"simulated": True, "truth_z": truth.z},
    )
    source = SourceContext(
        object_id=truth.object_id, ra=truth.ra, dec=truth.dec, tile_id=999999999, release="sim"
    )
    return SpectralObservation(source=source, combined=combined, dithers=tuple(dithers))


# --- FITS writer -----------------------------------------------------------
def _signal_hdu(
    spectrum,
    extname: str,
    fscale: float,
    extra: dict | None = None,
    with_ndith: bool = False,
) -> fits.BinTableHDU:
    cols = [
        fits.Column(name="WAVELENGTH", format="1E", unit="Angstrom", array=spectrum.wavelength),
        fits.Column(
            name="SIGNAL",
            format="1E",
            unit="erg/s/cm2/Angstrom",
            array=np.asarray(spectrum.flux) / fscale,
        ),
        fits.Column(name="MASK", format="1J", unit="Number", array=spectrum.mask),
        fits.Column(name="QUALITY", format="1E", unit="Number", array=spectrum.quality),
        fits.Column(
            name="VAR",
            format="1E",
            unit="erg2/s2/cm4/Angstrom2",
            array=np.asarray(spectrum.variance) / fscale**2,
        ),
    ]
    if with_ndith and getattr(spectrum, "ndith", None) is not None:
        cols.append(fits.Column(name="NDITH", format="1I", unit="Number", array=spectrum.ndith))
    hdu = fits.BinTableHDU.from_columns(cols, name=extname)
    h = hdu.header
    h["WMIN"] = (float(spectrum.wavelength[0]), "[Angstrom] Minimum wavelength of the binning")
    h["BINWIDTH"] = (float(spectrum.bin_width), "[Angstrom] Wavelength bin width")
    h["BINCOUNT"] = (int(spectrum.n_pixels), "Wavelength bin count")
    h["FSCALE"] = (fscale, "Scaling factor")
    h["EXPTIME"] = (float(getattr(spectrum, "exposure_time", np.nan)), "Exposure time")
    h["LSF_SIG"] = (float(spectrum.lsf_sigma), "[Angstrom] Std dev of the Gaussian LSF model")
    h["EXT_PROF"] = (str(getattr(spectrum, "extraction_profile", "NONE")), "1D extraction profile")
    for key, value in (extra or {}).items():
        h[key] = value
    return hdu


def write_sir_file(
    path: str | Path,
    observations,
    tile_id: int = 999999999,
    fscale: float = DEFAULT_FSCALE,
    overwrite: bool = True,
) -> Path:
    """Write simulated observations in the real SIR combined-spectra layout."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    primary = fits.PrimaryHDU()
    ph = primary.header
    ph["FITS_DEF"] = "sir.combinedSpectra"
    ph["FITS_VER"] = "0.2"
    ph["TELESCOP"] = "EUCLID"
    ph["INSTRUME"] = "NISPsim"
    for name, bit in MASK_BITS.items():
        ph[f"HIERARCH MSK_FLAG_{name}"] = (bit, "Mask bit position")
    ph["ORIGIN"] = "euclid_agn.validation.simulator"
    ph["TILE_ID"] = tile_id
    ph["N_OBJ"] = len(observations)
    ph["LRANGE"] = "RGS"
    hdus = [primary]
    for index, obs in enumerate(observations):
        meta = fits.ImageHDU(name=f"{index}_META")
        meta.header["N_DITH"] = (obs.n_dithers, "Number of dither containing object")
        meta.header["OBJ_ID"] = (obs.object_id, "MER-specified identifier")
        meta.header["RA_OBJ"] = (obs.source.ra, "[deg] R.A. of the m-th object")
        meta.header["DEC_OBJ"] = (obs.source.dec, "[deg] Declination of the m-th object")
        hdus.append(meta)
        hdus.append(
            _signal_hdu(
                obs.combined, f"{index}_COMBINED1D_SIGNAL", fscale, with_ndith=True
            )
        )
        for dither in obs.dithers:
            extra = {
                "DITH_ID": (dither.dither_id, "The dither index"),
                "PTGID": (dither.pointing_id, "Pointing ID in the name"),
                "DET_ID": (dither.detector_id, "The detector index"),
                "GWA_POS": (dither.gwa_position, "GWA position"),
                "GWA_TILT": (dither.gwa_tilt, "Commanded Grism Tilt Angle"),
            }
            name = f"{index}_DITH1D_{dither.pointing_id}_SIGNAL"
            hdus.append(_signal_hdu(dither, name, fscale, extra=extra))
            contam = fits.BinTableHDU.from_columns(
                [
                    fits.Column(
                        name="OBJ_ID",
                        format="1K",
                        array=np.asarray(dither.contaminant_object_ids, dtype=np.int64),
                    )
                ],
                name=f"{index}_DITH1D_{dither.pointing_id}_CONTAMINANTS",
            )
            for key, value in extra.items():
                contam.header[key] = value
            hdus.append(contam)
    fits.HDUList(hdus).writeto(path, overwrite=overwrite)
    return path


def make_test_file(path: str | Path, n_objects: int = 3, seed: int = 20260916) -> Path:
    """A small, deterministic SIR-format file for the test suite."""
    rng = np.random.default_rng(seed)
    observations = []
    for i in range(n_objects):
        broad = i == 1
        lines = [
            LineTruth("Halpha", 4.0e-16, 120.0),
            LineTruth("NII6584", 1.2e-16, 120.0),
            LineTruth("NII6548", 0.4e-16, 120.0),
        ]
        if broad:
            lines.append(LineTruth("Halpha", 1.2e-15, 2500.0, broad=True))
        observations.append(
            simulate_observation(
                SpectrumTruth(
                    z=1.20 + 0.05 * i,
                    lines=tuple(lines),
                    object_id=2_700_000_000_000_000_000 + i,
                    seed=int(rng.integers(1, 1 << 31)),
                    n_dithers=3 + (i % 2),
                    masked_fraction=0.05 * i,
                )
            )
        )
    return write_sir_file(path, observations)
