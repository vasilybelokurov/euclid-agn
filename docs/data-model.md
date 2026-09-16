# Data model

## Canonical units

| quantity | unit |
|---|---|
| wavelength | Å (vacuum) |
| flux density | erg s⁻¹ cm⁻² Å⁻¹ |
| variance | (erg s⁻¹ cm⁻² Å⁻¹)² |
| velocity | km s⁻¹ |

Conversion from archive units happens exactly once, in `euclid_agn.io.sir`.
Nothing downstream re-scales.

## SIR combined-spectra file layout (verified)

```
PRIMARY                           FITS_DEF='sir.combinedSpectra', TILE_ID, N_OBJ,
                                  LRANGE, MSK_FLAG_* bit definitions
<k>_META                          OBJ_ID, RA_OBJ, DEC_OBJ, N_DITH
<k>_COMBINED1D_SIGNAL             WAVELENGTH SIGNAL MASK QUALITY VAR NDITH
<k>_DITH1D_<PTGID>_SIGNAL         WAVELENGTH SIGNAL MASK QUALITY VAR
<k>_DITH1D_<PTGID>_CONTAMINANTS   OBJ_ID of overlapping sources
```

`<k>` is the object index *within the file*, not the object id. The number of
HDUs per object varies with the number of contributing dithers, so groups are
discovered from EXTNAME. A Q1 tile file holds 1000 objects in ~10 000 HDUs and
is ~112 MB; a single object costs tens of kB over lazy S3 range requests.

Per-spectrum header keywords used by the science model: `FSCALE`, `EXPTIME`,
`LSF_SIG`, `EXT_PROF`, `WMIN`, `BINWIDTH`, `BINCOUNT`, and for dithers
`DITH_ID`, `PTGID`, `DET_ID`, `GWA_POS`, `GWA_TILT`.

## Flux scaling

```
flux     = SIGNAL * FSCALE
variance = VAR    * FSCALE**2
```

Pinned by `tests/regression/test_real_q1_fixture.py`. The quadratic convention
was confirmed empirically: in raw stored units the pixel-to-pixel scatter of
SIGNAL matches √VAR (median ratio 1.25 over 973 spectra); the alternative
convention would misplace it by 16 orders of magnitude.

## Mask bits

`GOOD 0, NOT_USE 1, LOW_SNR 2, EXT_PRB 4, HIGH 8, LOW 16, REL_FLUX 32,
ABS_FLUX 64`, read from the primary header, used as a bit field. Observed
frequencies in tile 102160339: NOT_USE 9.5 %, LOW_SNR 32.3 %, EXT_PRB 0.6 %,
HIGH 1.2 %, LOW 0.1 %, REL_FLUX 0.0 %, ABS_FLUX 10.5 %.

Only `NOT_USE` is rejected by default. `LOW_SNR` pixels carry information that
the variance already down-weights; discarding them would throw away a third of
every spectrum. All fractions are recorded per spectrum as selection-function
inputs.

## Objects

`Spectrum1D` → `DitherSpectrum` / `CombinedSpectrum`, plus `SourceContext` and
`SpectralObservation`. All frozen; arrays are read-only. A Euclid spectrum is
not wavelength + flux: mask, variance, quality, LSF, dithers and contaminants
are part of the datum.

## Manifest

`data/manifests/*.parquet`, one row per source with a Q1 spectrum:
archive address (`sir_s3_key`, `sir_hdu`), MER photometry and morphology, PHZ
redshifts, SPE class/redshift/quality. Every catalogue column is context for
stratifying the selection function, never an AGN-selection input.
