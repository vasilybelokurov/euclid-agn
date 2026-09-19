# JOURNAL

## 2026-09-16 — session 1: package foundation (M0) and Q1 archive layer (M1)

### Project brief

Build a reproducible pipeline to find AGN in public Euclid NISP slitless spectra
by emission-line decomposition in the spirit of Chilingarian et al. (2018),
selecting on the spectra alone — no cut on host mass, luminosity, colour,
morphology, star-formation state or existing AGN classification — and
publishing the selection function against redshift, line flux, line width,
source size, host properties, spectral quality and contamination.

Working rules for this project:

- reusable Python in `src/`, never science-critical code in notebooks only;
- `JOURNAL.md` kept current with concrete provenance and decisions;
- no conclusions before tests are written **and run**.

### Workspace state at start

- Working directory `/Users/vasilybelokurov/Work/Code/euclid_spectral_fit`,
  which is a symlink into `~/IoA Dropbox/Dr V.A. Belokurov/Code/`. Empty.
- Not a git repository, and none was initialised (that is the user's call).
  `euclid_agn.provenance.git_commit()` returns `None` and the run manifest
  records that honestly.
- Environment `~/Work/venvs/.venv`, Python 3.13.5. Present: numpy 1.26.4,
  scipy 1.16.1, astropy 7.1.0, astroquery 0.4.10, fsspec/s3fs 2025.7.0,
  pyarrow 21, pandas 2.3.2, pydantic 2.11.7, typer 0.20, matplotlib 3.10.3,
  pytest 9.1.1, pyyaml. Absent: `uv`, `ruff`, `hypothesis`, `polars` — declared
  as optional dev extras rather than installed into the shared venv.
- Related existing work reviewed: `../qso_double_peaked/scripts/` (DESI
  narrow-line fitting with shared Δv, `FamilySpec`/`WindowSpec` dataclasses,
  `scipy.optimize.least_squares`, ΔBIC ranking). Its line-family dataclass
  pattern and manifest→per-object→table workflow are carried over; its
  purely-parametric Gaussian model is not, because this project needs the
  linear/non-linear split and the non-parametric NLR profile.

### Archive reconnaissance — what was measured, not assumed

All of the following was checked against the live IRSA service and real Q1
products on 2026-09-16.

**Tables.** `TAP_SCHEMA` lists the expected Q1 tables. Column names for MER,
MER morphology, PHZ photo-z, SPE classification/quality/candidates and the
object↔spectrum association table were read from the service rather than from
documentation.

**Association table.** `euclid.objectid_spectrafile_association_q1` maps
`objectid` → `path`, `hdu`. For tile 102160339 it holds 121 547 rows but only
18 251 have a non-null `path`. `path IS NOT NULL` is the spectrum-availability
filter. `path` is an IRSA web-API URL; the S3 key is recovered from it by
`sir_s3_key()`.

**Two TAP constraints found the hard way.**

1. MER and the association table live in different datasources; a server-side
   join is rejected (`unknown datasource / table`). Queries are therefore
   issued separately and joined client-side.
2. `SELECT position_angle FROM euclid_q1_mer_catalogue` is rejected with
   `BAD_REQUEST: Invalid or unsupported ADQL query string` — the parser trips
   on the reserved word `POSITION`. `SELECT "position_angle"` works. Quoting
   the whole SELECT list works; quoting identifiers in the WHERE clause breaks
   the query again. `quote_columns()` implements exactly that rule.

**Field footprints.** `SELECT DISTINCT tileid FROM euclid_q1_mer_catalogue
WHERE ra BETWEEN ... AND dec BETWEEN ...` failed after 5 minutes. Resolving a
field through the CAOM tables instead —
`euclid.tileid_association_q1 JOIN euclid.plane_euclid_q1 ON obsid` with
`CONTAINS(p.pt, CIRCLE(...))` — returns in ~15 s. Cone radii were then checked
for completeness:

| field | centre (deg) | radius | tiles |
|---|---|---|---|
| EDF-N | 269.73, +66.018 | 6.0° | 124 |
| EDF-S | 61.241, −48.423 | 6.0° | 148 |
| EDF-F | 52.932, −28.088 | 5.0° | 72 |
| LDN1641 | 85.4, −8.0 | 4.0° | 8 |

124 + 148 + 72 + 8 = 352, and `SELECT COUNT(DISTINCT tileid)` over the whole
association table is also 352. The four cones partition Q1 exactly.

**SIR file structure** (from
`q1/SIR/102160339/EUC_SIR_W-COMBSPEC_102160339_2024-11-05T16:26:34.614296Z.fits`,
112 MB, 9949 HDUs, `N_OBJ = 1000`, opened lazily over S3 in 8.4 s):

```
PRIMARY                           FITS_DEF='sir.combinedSpectra', TILE_ID, N_OBJ,
                                  LRANGE='RGS', MSK_FLAG_* definitions
<k>_META                          OBJ_ID, RA_OBJ, DEC_OBJ, N_DITH
<k>_COMBINED1D_SIGNAL             WAVELENGTH SIGNAL MASK QUALITY VAR NDITH
<k>_DITH1D_<PTGID>_SIGNAL         + DITH_ID, PTGID, DET_ID, GWA_POS, GWA_TILT
<k>_DITH1D_<PTGID>_CONTAMINANTS   OBJ_ID list of overlapping sources
```

`<k>` is the index of the object inside the file, not its OBJ_ID, and the HDU
count per object varies with dither count — so groups are discovered from
EXTNAME, never assumed. The association table's `hdu` value indexes the
`COMBINED1D_SIGNAL` extension directly.

**Wavelength grid.** 531 bins of 13.4 Å from 11900 Å to 19002 Å
(`WMIN`/`BINWIDTH`/`BINCOUNT`). The science-usable red-grism interval is
12500–18500 Å.

**FSCALE convention.** `FSCALE = 1e-16`. The DPDD says it applies to SIGNAL and
must be accounted for in VAR but does not state the power. Measured on the 973
usable spectra of the tile: the robust adjacent-pixel scatter of SIGNAL in raw
stored units, divided by the median √VAR in raw stored units, has percentiles
[5, 16, 50, 84, 95] = [0.99, 1.08, 1.25, 1.62, 2.19]. A ratio of order 1 means
VAR is stored in the same scaled units as SIGNAL, so

```
flux = SIGNAL * FSCALE          variance = VAR * FSCALE**2
```

The alternative convention would have put this ratio at 1e±16. Pinned by
`tests/regression/test_real_q1_fixture.py`. The median of 1.25 rather than 1.00
is itself a flag: either real spectral structure enters the adjacent-pixel
difference, or the variances are slightly optimistic. Either way it supports
the decision to calibrate detection significance empirically.

**Mask bits.** Definitions carried in the primary header behave as a bit field
despite the "Mask bit position" comment: observed MASK values include 65 = 64|1
and 67 = 64|2|1. Frequencies over the tile: NOT_USE 9.5 %, LOW_SNR 32.3 %,
EXT_PRB 0.6 %, HIGH 1.2 %, LOW 0.1 %, REL_FLUX 0.0 %, ABS_FLUX 10.5 %. Default
rejection is NOT_USE only; rejecting LOW_SNR would discard a third of every
spectrum for information the variance already carries.

**LSF_SIG.** Percentiles [1, 5, 25, 50, 75, 95, 99] over the 1000 objects:
[12.3, 12.9, 13.3, 13.7, 15.6, 54.1, 74.1] Å. So R = λ/(2.355σ) ≈ 465 at
1.5 µm for compact sources, with an extended-source tail down to R ≈ 90. The
median σ of 13.7 Å against a 13.4 Å bin is about **one sigma per pixel**, so
model lines must be integrated across pixels, not sampled at bin centres — the
peak bias is ~0.3 % for a 13.7 Å line and grows for narrower ones.

**Rest-wavelength convention.** Taking 2000 Q1 SPE H-α detections with
`spe_line_snr_gf > 10` and rank-0 galaxy redshifts (400 matched), the implied
rest wavelength λ_obs/(1+z) has median **6564.39 Å** (IQR 6559.5–6569.8).
Vacuum H-α is 6564.61 Å, air is 6562.80 Å. The line catalogue is therefore
vacuum throughout. The ±10 Å scatter (±450 km s⁻¹) means one line does not
settle it, but the median is 0.2 Å from vacuum and 1.6 Å from air.

**Reference object 2731173428682078045** (the IRSA cloud-access tutorial
target): tile 102160339, HDU 1644, RA 273.1173, Dec +68.2078, 4 dithers
(pointings 11889–11892, GWA RGS000/RGS180 alternating, detector 32), combined
`EXPTIME` 2198.569 s vs 549.6422 s per dither, contaminants 9/9/13/11 across
dithers. It is *not* a typical source: `LSF_SIG` = 98.4 Å (R ≈ 65),
`segmentation_area` = 110 334 pixels, aperture H fluxes null. It is a large
extended galaxy, useful as an IO test and a worked example of the LSF tail, not
as a model-development template. A compact companion fixture
(2734482961680140786, `LSF_SIG` = 12.5 Å, median per-pixel S/N 160) was frozen
alongside it.

**The tutorial reference object is almost entirely masked.** Running the
pipeline on 2731173428682078045 end-to-end showed that 98.3 per cent of its
combined-spectrum pixels carry `NOT_USE`. The nine survivors sit at
18881-19002 Angstrom, i.e. outside the 12500-18500 Angstrom science window, and
their apparent per-pixel S/N of 411 is a red-edge artefact; the median `QUALITY`
of the spectrum is 0.0. The compact companion 2734482961680140786 has a usable
fraction of 0.989 and median `QUALITY` 0.978. Two consequences:

- a selection on signal-to-noise alone would have accepted a spectrum with nine
  usable pixels, so `usable_pixel_fraction` (default threshold 0.5 in
  `configs/q1.yaml`) is a required parent-sample criterion, not a nicety;
- the first online test written for this object asserted `> 300` usable pixels
  and failed. The assertion was wrong, not the data. It now asserts the real
  behaviour, and `tests/regression/test_real_q1_fixture.py` pins the contrast
  between the two objects.

### Decisions

- **Package layout** follows the brief: `src/euclid_agn/` with `archive/`,
  `io/`, `spectra/`, `models/`, `fit/`, `validation/`, `pipeline/`, `plotting/`.
  Installed-free operation via `PYTHONPATH=src`; `pip install -e .` gives the
  `euclid-agn` console script. The shared venv was deliberately not modified.
- **Units are converted exactly once**, in `io/sir.py`. Everything downstream is
  plain NumPy in canonical units.
- **Domain objects are frozen and their arrays read-only**, so a fitting routine
  cannot quietly modify a spectrum.
- **Masked pixels are dropped, never interpolated.** `usable()` also drops
  non-finite flux and non-positive variance.
- **All quality/contamination statistics are recorded, not applied**, because
  they are the inputs to the selection-function analysis later.
- **Own minimal TAP client** rather than astroquery: a `/sync` POST is stable
  across releases, trivially mockable in unit tests, and logs the exact ADQL for
  provenance.
- **Manifest work is sharded per tile and skips existing shards**, so a
  field-scale build is restartable.
- **Regression fixture is real data**: a 222 kB two-object extract of the real
  tile file, headers and values unchanged, only the `<k>_` prefixes renumbered.
  Unit and regression tests need no network; online tests are opt-in.

### Files created

```
pyproject.toml                       configs/q1.yaml, configs/validation.yaml
README.md  JOURNAL.md
docs/{project-spec,data-model,science-model,validation}.md
src/euclid_agn/__init__.py  constants.py  config.py  provenance.py  cli.py  __main__.py
src/euclid_agn/archive/{base,schema,tap,irsa,esa}.py
src/euclid_agn/io/sir.py
src/euclid_agn/spectra/{types,masking,lsf}.py
src/euclid_agn/models/line_catalog.py
src/euclid_agn/validation/simulator.py
src/euclid_agn/pipeline/ingest.py
src/euclid_agn/plotting/diagnostics.py
tests/conftest.py
tests/unit/{test_masking,test_lsf,test_line_catalog,test_types,test_sir_io,
            test_simulator,test_archive,test_config_provenance}.py
tests/regression/test_real_q1_fixture.py
tests/online/test_irsa_reference_object.py
tests/data/EUC_SIR_W-COMBSPEC_102160339_two_objects.fits   (222 kB, real data)
```

### Run commands

```bash
source ~/Work/venvs/.venv/bin/activate
cd ~/Work/Code/euclid_spectral_fit
export PYTHONPATH=src

python -m pytest tests -q                      # unit + regression, offline
python -m pytest tests/online -m online -q     # opt-in, hits IRSA
python -m euclid_agn spectra inspect --object-id 2731173428682078045 \
    --plot outputs/reference_object.png
python -m euclid_agn archive build-manifest --config configs/q1.yaml \
    --tile 102160339 --limit 200 --output data/manifests/q1_tile102160339.parquet
```

### Test results

- Offline suite: **90 passed, 5 deselected** (unit + regression; the deselected
  five are the online tests), runtime 0.7 s.
- Online suite: **5 passed in 182 s** against the live IRSA service. It checks
  object lookup, lazy S3 read of the right object, manifest context columns,
  field-to-tile resolution (EDF-N = 124 tiles, LDN1641 = 8, reference tile in
  EDF-N) and the null-path behaviour of the association table.
- End-to-end manifest build, tile 102160339, `--limit 200`: 200 sources x 71
  columns in 93 s, with `run_manifest.json` written alongside. SPE class
  breakdown of that subset: 134 galaxy, 24 star, 19 QSO, 23 unclassified;
  174 have a rank-0 SPE galaxy redshift. This is a data-availability sample,
  not a science sample.
- `spectra inspect --object-id 2731173428682078045 --plot ...` produced the
  per-dither diagnostic figure and the metrics quoted above.

### Observation worth following up

The per-dither panels of the compact fixture object (2734482961680140786) show
narrow one- to two-pixel spikes of order 50 per cent of the continuum that
differ between dithers and are largely absent from the co-added spectrum.
Whatever they are - contamination from the 3-8 overlapping sources per dither,
cosmic rays, or detector artefacts - they are exactly the failure mode the
Stage-2 dither-coherence test is meant to catch, and they are visible before any
fitting. Worth quantifying: what fraction of apparent narrow features in
individual dithers survive co-addition, as a function of contaminant count.

### Next actions

1. M2: constrained linear solver (non-negative amplitudes, smoothness-penalised
   non-parametric NLR profile), B-spline continuum basis, broad-component
   basis — each with synthetic recovery tests before any real-data fit.
2. Property-based tests once `hypothesis` is available: flux conservation
   through convolution, monotonic broad-flux recovery, dither-permutation
   invariance of the joint likelihood.
3. Verify `QUALITY` semantics against a Euclid document and decide whether it
   enters the weights.
4. Quantify pixel-to-pixel covariance in line-free regions of real spectra; it
   bounds how much a Δχ² can ever be trusted.
5. Only then M3: Stage-1 combined-spectrum screening on a curated EDF-N subset.

---

## 2026-09-16 — session 2: public repo, data location, spectra availability, M2 primitives

### Repository

Public GitHub repository created at
https://github.com/vasilybelokurov/euclid-agn (initial commit `61c0b06`).
`euclid_agn.provenance.git_commit()` now returns a real SHA, so run manifests
are reproducible as the spec requires.

### Data location

User instruction: downloaded data go to `~/data/euclid/`. That directory already
held other Euclid work (DR1 star catalogues, Q1 cutouts and manifests, and
`with_desi/desi_euclid_q1_galaxies.fits`, which will matter for M4 truth
ingestion). `euclid_agn.io.cache.ArchiveCache` mirrors the archive's own layout
underneath it, so SIR products land at

```
~/data/euclid/q1/SIR/<tile>/EUC_SIR_W-COMBSPEC_<tile>_<timestamp>.fits
```

alongside the existing `q1/cutouts`, `q1/manifests`, `q1/mosaics`. Downloads are
atomic (temporary file, then rename), so an interrupted transfer can never be
mistaken for a complete one. The cache is on by default; `archive.cache_enabled:
false` streams byte ranges instead, which is cheaper for one object and much
more expensive for a whole tile.

### Spectra availability — measured

**Catalogue level.** Counts from `euclid.objectid_spectrafile_association_q1`:

| field | tiles | association rows | with a spectrum | fraction |
|---|---:|---:|---:|---:|
| EDF-N | 124 | 11 378 352 | 1 683 630 | 14.8 % |
| EDF-S | 148 | 13 060 965 | 1 874 296 | 14.4 % |
| EDF-F | 72 | 5 328 489 | 749 251 | 14.1 % |
| LDN1641 | 8 | 185 624 | **0** | 0 % |
| **total** | **352** | **29 953 430** | **4 307 177** | **14.4 %** |

The rows and the spectra both sum exactly to the Q1 totals queried
independently, which re-confirms that the four field cones partition the
release. Two things worth carrying forward:

- 4 307 177 matches the "about 4.3 million attempted extractions" in the
  project brief, so the brief's parent-sample figure is confirmed against the
  archive rather than taken on trust;
- **LDN1641 has no slitless spectroscopy in Q1 at all.** The star-forming field
  contributes 185 624 MER sources and zero spectra. Any survey-wide statement
  must be about EDF-N, EDF-S and EDF-F only.

**Pixel level.** 1214 combined spectra opened across seven tiles in the three
spectroscopic fields (`euclid-agn archive availability --sample-tile ...`,
figure `outputs/availability_sample.png`, table
`outputs/availability_sample.parquet`):

| quantity | value |
|---|---|
| usable pixel fraction in 12500-18500 Å, median | 0.97 |
| fraction with usable fraction ≥ 0.5 | 79.1 % |
| fraction with **zero** usable science pixels | 13.6 % |
| median continuum S/N per pixel (fittable spectra) | 3.1 |
| fraction fittable **and** S/N > 3 | 40.4 % |
| fraction fittable **and** S/N > 1 | 58.7 % |
| effective LSF σ, median / p90 / p99 | 15.0 / 25.6 / 61.9 Å |
| fraction with LSF σ above twice the median | 5.9 % |
| fraction with 4 or more contributing dithers | 58 % |
| fraction with 2 or fewer dithers | 24 % |

Scaling the sample fractions to the 4.31 million available spectra gives a
working expectation of roughly 3.4 million with enough unmasked pixels to fit,
about 2.5 million with continuum S/N above 1, and about 1.7 million above S/N 3.
These are availability numbers, not detectability numbers — what a given S/N
buys in broad-line sensitivity is an injection/recovery question for M4.

Consequences for the design:

- the dither-coherence discriminator, which the plan leans on heavily, is only
  strong for the 58 % with four dithers; for the 24 % with one or two dithers it
  barely exists, and that has to be a reported axis of the selection function,
  not a silent cut;
- 13.6 % of extracted spectra carry no usable science pixels at all, so
  "has a spectrum" and "can be fitted" differ by a factor of about 1.2 before any
  S/N consideration.

### M2 progress — model primitives

Delivered with synthetic tests: constrained regularised linear solver, B-spline
continuum, non-parametric narrow-line system, broad-line components, and the
forward model that assembles M0/M1.

**Two real bugs found by the tests, both now fixed.**

1. *The regularisation weight was silently ignored.* The solver normalised the
   penalty matrix to the Frobenius norm of the design matrix, which discarded
   the caller's strength entirely: smoothness 0.3 and 1.0 gave bit-identical
   fits. Penalty *structure* (a difference operator) and *strength* (a
   dimensionless weight) are now separate, and a test asserts that varying the
   weight monotonically changes the effective degrees of freedom.

2. *The alternating narrow-line fit is not exactly monotone with a penalty.*
   With the scale-free penalty the objective is re-normalised at each half-step,
   so the data chi-squared can rise slightly between iterations — measured at
   0.03 % of chi-squared. The unpenalised case is exactly monotone, as theory
   requires. Both behaviours are now separate tests rather than one wrong
   assertion.

**Numerical note.** numpy 1.26.4 built against Apple Accelerate raises spurious
`divide by zero / overflow / invalid value encountered in matmul` warnings on
ordinary finite matrix products; verified on a 600×64 by 64×600 product of
Gaussian random numbers. `euclid_agn.numerics` suppresses them around
matmul-heavy blocks and asserts finiteness of the results instead, so a real
numerical failure still raises.

**Design decision recorded.** The narrow-line problem is *bilinear*, not linear:
the model is linear in the amplitudes at fixed profile and linear in the profile
at fixed amplitudes, but not in both at once. The brief's phrase "solve
amplitudes/profile coefficients as a constrained regularised linear problem"
holds for each half. The implementation alternates between the two constrained
solves with the continuum re-fitted in both, fixes the scale degeneracy by
normalising the profile to unit integral so amplitudes are physical fluxes, and
reports its chi-squared history. A test confirms the answer is insensitive to
the starting profile at the 5 % level.

**Resolution caveat.** One bin is 13.4 Å and the effective LSF σ is about the
same, so at H-α in mid-range one pixel is roughly 270 km/s. A velocity grid
finer than that measures the penalty, not the data;
`suggested_velocity_step` reports the resolution-matched value.

### Test results

- Offline suite: **154 passed, 5 deselected**, 1.2 s.
- Online suite: **5 passed in 182 s** (run earlier this session, unchanged).

Headline synthetic results now in the suite:

- a noiseless pure-narrow spectrum yields a broad flux below 5 % of the narrow
  H-α flux — the model does not manufacture a BLR;
- an injected broad H-α of 1.5e-15 erg s⁻¹ cm⁻² at 2500 km s⁻¹ is recovered to
  25 % with Δχ² > 100;
- both Δχ² and the recovered broad flux increase monotonically with injected
  broad flux;
- a broad component on a forbidden transition is refused unless explicitly
  requested for a null test.

### Next actions

1. `fit/hypotheses.py`: redshift-hypothesis generation from SPE, PHZ, detected
   peaks and a blind line-family scan, with the origin of each hypothesis
   recorded.
2. `fit/screen.py`: Stage-1 scan over (z, σ_BLR, Δv_BLR) with the small
   non-linear space optimised and the linear solve inside.
3. Measure pixel-to-pixel covariance in line-free regions of real spectra before
   any Δχ² is turned into a significance.
4. First Stage-1 run on a curated EDF-N subset, with DESI labels from
   `~/data/euclid/with_desi/desi_euclid_q1_galaxies.fits` held back.

---

## 2026-09-16 — session 3: noise model measured, ID convention corrected

### The Q1 noise model is not what a naive chi-squared assumes

Q1 supplies a per-pixel variance and no covariance, and SIR resamples dispersed
2D data onto a common 1D grid, which must correlate neighbouring pixels.  Both
effects were measured on **934 real combined spectra** from eight cached tile
files across EDF-N, EDF-S and EDF-F (`euclid-agn spectra noise-audit`,
table `outputs/noise_audit.parquet`).

Method: fit a smooth B-spline continuum on the longest contiguous run of usable
pixels inside 12500-18500 Angstrom, form `r = (f - continuum) / sqrt(VAR)`,
clip beyond 4 sigma so emission lines are not counted as noise, then take the
standard deviation and the autocorrelation of what remains.

The procedure has a bias of its own - a continuum with p free parameters
absorbs part of the noise - so it was **calibrated on simulated white noise with
exact variances** (`measure_method_bias`), and the bias divided out.  The
control recovers `sigma_r = 0.963`, `rho_1 = -0.060` at 25 knots instead of the
ideal 1.0 and 0.0, confirming the bias is small and downward.

Bias-corrected results, as a function of how flexible the continuum is allowed
to be (531 pixels on the full grid, ~450 usable):

| interior knots | pixels per knot | sigma_r | rho_1 | inflation eta | chi2 scale eta^2 |
|---:|---:|---:|---:|---:|---:|
| 8 | ~56 | 1.265 | +0.229 | 1.53 | 2.33 |
| 12 | ~37 | 1.241 | +0.219 | 1.49 | 2.22 |
| 25 | ~18 | 1.204 | +0.184 | 1.41 | 1.98 |
| 50 | ~9 | 1.147 | +0.128 | 1.29 | 1.65 |

Reading the table honestly: some of the excess is genuine smooth structure that
a stiff continuum fails to follow, which is why the numbers fall as knots are
added; but 50 knots on 450 pixels is already absorbing real noise, so the truth
is bracketed rather than pinned.  At the defensible middle of the range:

> **Reported variances are too small by a factor of about 1.45, adjacent pixels
> are positively correlated at rho_1 = +0.18, and a naive Delta chi-squared
> overstates the evidence for an added component by a factor of about 2
> (bracketed 1.65-2.33).**

The spread between objects is larger than the spread between methods: the 84th
percentile of the per-spectrum inflation is 1.78, i.e. a chi-squared scale of
3.2.  Lags of 2 and beyond come out slightly negative (-0.07 to -0.11), the
expected signature of having removed a smooth component, not evidence of
anti-correlated noise.

Consequences, which were already the plan but are now quantified:

- a Delta chi-squared of 25 - a naive "5 sigma" - corresponds to an effective
  value near 12, i.e. roughly 3.5 sigma, *before* accounting for the fact that
  the broad flux is bounded at zero and many hypotheses are scanned per object;
- because the inflation varies by object, a single global correction is not
  enough; the per-spectrum inflation belongs in the candidate table as a
  covariate, and final significances still have to come from empirical nulls;
- a fitted noise-scale nuisance parameter in the likelihood is justified by
  measurement, not by taste.

### Object identifiers are signed, and negative in the south

Auditing the cached files turned up whole tiles whose `OBJ_ID` values are
negative, e.g. -638864563487453476.  This is **not** corruption.  The MER
identifier encodes the source position:

```
 2731173428682078045  ->  RA 273.1173428  Dec +68.2078045   (EDF-N)
-638864563487453476   ->  RA  63.8864563  Dec -48.7453476   (EDF-S)
```

Sources south of the equator carry negative identifiers, consistently in the
FITS headers and in the TAP catalogues (checked against both
`euclid.objectid_spectrafile_association_q1` and `euclid_q1_mer_catalogue` for
tile 102021017).  Since EDF-S and EDF-F together hold 2 623 547 of the
4 307 177 Q1 spectra, *the majority of the parent sample has a negative object
id*.  Any code that validates an id by requiring it to be positive, or stores
one in an unsigned column, will silently drop most of the survey.  A unit test
now round-trips a southern identifier, and the earlier test that asserted
`object_id > 0` has been corrected.

### N_OBJ is a nominal value

Also verified: the primary-header `N_OBJ` is not the number of objects in the
file.  One Q1 file for tile 102160339 carries `N_OBJ = 1000` and holds 1000
object groups; another file of the same tile carries `N_OBJ = 1000` and holds
159.  `SirCombinedSpectraFile.n_objects` now counts discovered groups, and
`n_objects_header` is retained for provenance only.

One outlier file was noted and set aside: the cached file for tile 102018211
was written by SIR 5.0.5 rather than 5.0.6, holds 5 object groups against
`N_OBJ = 465`, and every one of its spectra is completely masked.

### Test results

- Offline suite: **169 passed, 5 deselected**, 1.3 s.

### Next actions

1. `fit/hypotheses.py`: redshift hypotheses from SPE, PHZ, detected peaks and a
   blind line-family scan, each tagged with its origin.
2. `fit/screen.py`: Stage-1 scan over the small non-linear space, carrying the
   per-spectrum noise inflation into the recorded statistics.
3. First Stage-1 run on a curated EDF-N subset with DESI labels held back.

---

## 2026-09-16 — session 4: M3 Stage-1 screening, and what real spectra did to it

### Architecture: Stage 1 is itself two tiers

Cost decides the design.  Q1 has 4.3 million spectra and a blind scan at the
instrumental resolution puts ~1900 hypotheses on each.  Running the alternating
narrow-line fit at every hypothesis is impossible at that scale.

**Tier A, matched filter.**  The continuum basis does not depend on redshift, so
it is whitened and orthonormalised once per spectrum; each hypothesis is then a
few dot products after projecting its line basis orthogonal to the continuum
span.  Measured cost: **0.44 ms per hypothesis-row**, flat in the number of
hypotheses.

| blind step | hypotheses | time per spectrum | all Q1 |
|---:|---:|---:|---:|
| 2000 km/s | 282 | 0.13 s | ~150 core-hours |
| 1000 km/s | 561 | 0.25 s | ~300 core-hours |
| 300 km/s | 1868 | 0.81 s | ~970 core-hours |

Restricting to the ~3.4 million fittable spectra, a 300 km/s blind scan is about
770 core-hours — a day on 32 cores.  Feasible, so the blind scan stays.

**Tier B, full fit.**  Only the best few hypotheses get the M0/M1 treatment with
the non-parametric profile and bounded solves.

### Three failure modes, each found by running on real data

None of these came from reasoning about the method; each came from looking at
what the pipeline actually did to Q1 spectra.

**1. Ranking by broad-line gain alone gets the redshift wrong.**
H-alpha at z = 1.2 falls at 14442 Å; Pa-beta at z = 0.1264 falls at 14458 Å.
That is **1.15 pixels**.  On a synthetic object with a broad H-alpha, ranking by
broad gain put a single-line Pa-beta identification of the same feature in the
top five and the true redshift nowhere.  Ranking by narrow + broad evidence put
z = 1.2009 first, because the H-alpha system contributes six lines and the
Pa-beta system one.

Fix: the redshift is chosen on all the line evidence; the broad gain remains the
AGN statistic.  Both rankings are refined — the union of best-by-total and
best-by-broad — so an object whose only signal is a BLR is not discarded by a
rule designed for the opposite case.  Each row now records how much better the
winner is than the best hypothesis of a *different* line system.

**2. Every candidate railed at the widest width in the grid.**
The first real-data pilot (120 spectra) returned a top-eight in which every
broad FWHM was 5887 or 11774 km/s — the two largest in the grid — and the median
spectrum "preferred" a broad component, at an effective Δχ² of 6.2.

A broad Gaussian is degenerate with continuum curvature, and the degeneracy is
measurable: project the whitened broad basis vector orthogonal to the continuum
span and see what fraction of its norm survives.  On the real Q1 grid
(448 fitted pixels):

| σ (FWHM) km/s | 6 knots | 12 knots | 25 knots | 50 knots |
|---|---|---|---|---|
| 300 (706) | 0.95 | 0.92 | 0.86 | 0.73 |
| 600 (1413) | 0.93 | 0.86 | 0.78 | 0.56 |
| 1200 (2826) | 0.86 | 0.74 | 0.58 | 0.23 |
| 2500 (5887) | 0.70 | 0.47 | 0.21 | 0.01 |
| 5000 (11774) | 0.42 | **0.13** | 0.01 | 0.00 |
| 8000 (18839) | 0.19 | 0.02 | 0.00 | 0.00 |

At the working continuum flexibility of 12 knots, a σ = 5000 km/s line keeps
13 % of its norm; the fit is then driven by a small orthogonal remnant, which is
precisely how continuum mismatch becomes a "broad-line detection".  **A large
part of the classical BLR width range is intrinsically degenerate with the
continuum in the Euclid red grism.**  That is a property of the data.

Fix: widths failing `min_broad_orthogonality` are not scanned, and the
orthogonality of the best surviving width is recorded on every row as a
selection-function axis.  The median effective statistic over the same 117
spectra fell from 6.2 to 1.6.

**3. The best surviving candidate was an edge artefact.**
Plotting it settled the matter immediately: object 2726095024677601371, nominal
Δχ² = 214, was a broad profile centred at **18200 Å** — on the red edge, half of
it off the detector — while the M0 residuals were ±10σ across the whole range.

Fix: a broad component must now have at least 80 % of its flux on covered
pixels, and every row carries the reduced chi-squared of M0 so that a Δχ² drawn
from a failed continuum model is visible rather than implicit.

### An independent consistency check

After the guards, the median reduced chi-squared of M0 over real spectra is
**2.1**.  The noise audit of session 3, which knows nothing about this fit,
measured a chi-squared inflation of **η² = 1.98** from residual scatter and
pixel-to-pixel correlation alone.  The agreement says the continuum-plus-narrow
model is *adequate*, and that the apparent misfit is the reported variances
being too small rather than the model being wrong.  That is reassuring for the
decision not to start with a stellar-population continuum.

### Pilot results after all three guards

120 real spectra from two EDF-N tiles, blind scan at 600 km/s:

| | all 117 rows | the 80.3 % whose continuum model is acceptable |
|---|---|---|
| median effective Δχ² | 1.6 | 1.2 |
| 90th percentile | 17.6 | 9.3 |
| 95th percentile | 29.1 | 21.3 |
| 99th percentile | 71.1 | 57.2 |

These are essentially random spectra, most of which contain no AGN, so this
distribution is close to an empirical null — and it is wide.  A nominal
Δχ² = 25 sits near the 95th percentile of *unselected* spectra.  Any threshold
has to come from this distribution and from injection/recovery, not from a
chi-squared table.  Nothing here is called a detection.

### Files added

```
src/euclid_agn/fit/hypotheses.py      redshift hypotheses, four origins
src/euclid_agn/fit/screen.py          two-tier Stage 1
src/euclid_agn/pipeline/screening.py  manifest -> screening table
src/euclid_agn/pipeline/noise.py      (session 3) noise audit
tests/unit/test_hypotheses.py, test_screen.py, test_screening_pipeline.py
outputs/screen_pilot*.parquet, outputs/candidate_*.png
```

### Test results

- Offline suite: **221 passed, 5 deselected**, 3.7 s.

### Next actions

1. M4 validation: injection and recovery into real control spectra, and the
   empirical null.  The pilot distribution above is the first sketch of the
   null; it needs the proper construction with off-line redshifts, forbidden
   broad components and contamination-rich controls.
2. Ingest DESI truth from `~/data/euclid/with_desi/desi_euclid_q1_galaxies.fits`
   and partition it into development, validation and a blind holdout **before**
   any threshold is chosen.
3. Report completeness against the measured degeneracy axes — broad width,
   continuum orthogonality, effective LSF, usable fraction, dither count —
   rather than a single flux-completeness curve.

---

## 2026-09-16 — session 5: diagnostic atlas

`euclid-agn plot` writes a PNG atlas into `plots/`, regenerable from the
screening table plus the cached archive files:

```bash
python -m euclid_agn plot --screen outputs/screen_pilot_v3.parquet \
    --glob '~/data/euclid/q1/SIR/*/*.fits' --directory plots --n-candidates 8
```

| figure | content |
|---|---|
| `candidate_grid.png` | top candidates zoomed on the broad line under test |
| `screen_statistics.png` | the statistic's population distribution and what drives it |
| `availability_sample.png` | the parent sample: usable fraction, S/N, LSF, dithers |
| `fits/fit_<id>.png` | one candidate: data, M0, M1, broad-on-continuum, residuals |
| `spectra/spectrum_<id>.png` | combined spectrum above each contributing dither |

**A plotting bug worth recording.** The first version drew the broad component
on top of *M0's* continuum. That is not a decomposition of anything: M1 refits
the continuum and the narrow lines in the presence of the broad component, so
the two continua differ. The overlay is now M1 with its narrow block zeroed,
and a test asserts the invariant that overlay + narrow reconstructs M1 exactly.
On clean synthetic spectra the two versions agree closely, which is precisely
why it needed a structural test rather than a numerical one.

**What the figures show.** `screen_statistics.png` puts the median reduced
chi-squared of M0 (2.08) next to the independently measured noise inflation
(eta^2 = 1.98); the two lines nearly coincide. The recovered-width histogram is
still bimodal at the extremes of the allowed grid, so even after the
orthogonality cut some preference for the widest permitted width survives - the
guard reduced the problem rather than removing it, and the residual is exactly
what injection/recovery has to quantify.

Reading the individual fits remains the fastest way to find a failure mode: the
edge-artefact candidate of session 4 was obvious in one figure and invisible in
the numbers.

---

## 2026-09-16 — session 6: do the fits work? Blind redshift recovery against SPE

### The question and the test

"Do we have objects with reliable emission lines and useful fits?"  Answered
with an experiment rather than an opinion: give the pipeline **no catalogue
information**, let it scan redshift blind, and compare where it lands with
Euclid's own SPE redshift.  Agreement needs IO, masking, continuum, line list,
LSF, matched filter and ranking all to be right at once, against an independent
measurement of the same photons.  It says nothing about AGN; it says whether
lines are real and fits are useful.

Data in hand: **146 MB — 8 SIR files, 8 tiles, 1373 objects** (0.03 % of Q1).
Of these, 675 have an SPE rank-0 galaxy redshift and 93 have an SPE line at
S/N > 5: Hα 23, [O III] 5008 20, [S II] 6718 16, Hβ 7, [S III], Paschen.

### First result: 2.9 % agreement, and why

659 objects compared, tolerance 1000 km/s: **2.9 % agree** (16 % of the 31 with
an SPE line at S/N > 10).  A failure, but a diagnostic one.  75 % of objects were
assigned `hbeta_oiii`, the system with the most members (6.6 visible on
average).  Raw Δχ² with free non-negative amplitudes has no penalty for the
number of components, so the largest system wins.

The size of the bias was measured on 150 real spectra (best statistic within
each system over the full blind scan; most objects have no strong lines, so
this is close to a null):

```
median per-object max Δχ²  =  7.0 + 6.6 k       k = 1…8 free components
```

Applying that as a linear per-component penalty **made agreement worse**
(2.1 % → 1.1 %) and moved the dominant system to `paschen_beta` (41 %): the bias
relocated from the biggest system to the smallest.  A per-component penalty is
the wrong tool.

### The real causes, found by looking at the 31 strongest-line objects

Comparing the statistic at the SPE redshift against the winner's showed three
distinct failure classes.

**1. SPE false positives.**  Two "strong lines" are not there: the claimed
[S III] 9530 (S/N 13.8, FWHM 92 Å) sits on pixels reading 1.04, 0.97, 0.87,
0.89, 0.98, 0.86, 0.94 ×10⁻¹⁷ — flat; the claimed [O I] 6303 (S/N 43.5,
FWHM 143 Å) sits on noise at continuum S/N 1.2.  Real narrow lines have
FWHM ≈ LSF ≈ 32–60 Å.  Across the cached sample **17 % of SPE lines at S/N > 5
have FWHM > 80 Å** (22 of 93 objects): 6 of 22 [O III] 5008, 3 of 4 Pa-β,
3 of 4 [S III] 9530; Hα and [N II] are clean (0 of 50).  SPE line S/N is not
usable as truth without a width check; `spe_reference` now drops such lines.

**2. Multi-line system degeneracy at the coarse grid.**  Object
2688606391657706383 has Hβ and [O III] 5008 at S/N 21 and 44 — both clearly
present (my matched filter gives Δχ² 534 and 320 at the SPE wavelengths).  At
z = 1.2 the Hα complex maps Hβ→Hα and [O III]→[S II] 6731 to within 1.6 pixels,
and on a 600 km/s grid the wrong identification wins by 3 %.  The wavelength
*ratios* differ by 0.4 % and do discriminate — but only once each candidate is
placed at its own best redshift.  Tier B previously fitted at the fixed coarse
redshift.  `refine_redshifts_locally` now re-scans the best coarse candidate
of every system on a 60 km/s grid before the winner is chosen.

**3. List-order identification in overlapping windows.**  The blind grid emitted
one hypothesis per redshift with the *first* visible system.  Over
1.57 < z < 1.82 both Balmer systems are visible; the identification was being
decided by the order of `SYSTEMS`.  Fixed: one hypothesis per visible system.

A fourth class is real but deferred: bright low-redshift galaxies (e.g. the
Pa-β S/N 51 object at z = 0.015) whose stellar continuum has structure a
12-knot spline cannot follow, so everything looks like a line.  The reduced-χ²
gate flags them; identifying them needs a continuum that follows stellar
features, i.e. the stellar-population model the brief deferred until residuals
demanded it.  For this class, they do.

### Other work this session

- Stage-1 scan **2× faster with bit-identical output** (1.5 → 0.71 ms per
  hypothesis; max absolute difference 0.0 over 4664 rows): pixel edges computed
  once, broad columns reused between the orthogonality test and the fit.
- The measured per-object noise scale now **enters the likelihood**
  (`Spectrum1D.with_variance_scale`), and the inflation is split into its
  variance-scale and correlation parts so nothing is corrected twice.
- **DESI truth set sized** (`~/data/euclid/with_desi/desi_euclid_q1_galaxies.fits`):
  44 667 DESI–Euclid matches, all `GALAXY` spectype (no QSO class in this file),
  RA 262.5–277.0, Dec 63.1–68.9 (EDF-N).  With Δχ²_DESI > 25 and H ≲ 22.5:
  17 527 in the Pa-β window, 20 951 in He I/Pa-γ/δ, 9 284 in Hα, 254 in
  Hβ–Hδ.  A broad-line AGN label set will need a separate pull.

### Pending

Blind recovery re-run with the three fixes, three variants (raw ranking;
per-system null offsets; raw + SPE/PHZ hypotheses as production would use).
Results appended below when the run completes.

### Results of the fixes, and what the confident failures turned out to be

Blind recovery after local refinement, all-systems grid and the SPE width cut:
2.6 % (raw), 1.8 % (per-system offsets), 3.2 % (with SPE/PHZ hypotheses
added).  The last number is the decisive one: handed the correct redshift as
an explicit hypothesis, the scan still preferred a wrong one 97 % of the time.
The statistic, not the grid, was the problem.

**Fixed-ratio templates for identification.**  With one free non-negative
amplitude per line, a hypothesis is never penalised for predicting a line that
is absent — there is no evidence *against*.  `models/templates.py` gives each
system a few fixed-ratio templates; the identification statistic is one joint
fit of the best template plus the broad column, whatever the number of lines.
Summing separately optimised template and broad statistics was tried first and
let a Mg II doublet and a broad Gaussian both claim the same blob; the joint fit
counts shared flux once.  Templates choose the redshift only; measurement keeps
free amplitudes.  Synthetic Hβ+[O III] vs Hα-complex (1.6 px apart): template
statistic for the wrong identification < 0.7 of the right one.

Effect on real data: S/N > 5 objects 5.4 % → 12.2 %; overall unchanged at ~3 %.

**The bulk of the 659 has nothing to identify.**  Continuum S/N ≈ 3, no line
above S/N 5 in 566 of them.  A maximum over ~1000 noise hypotheses beats one
truth hypothesis by construction.  That is not a pipeline failure and the
denominator was wrong.  Restricted to objects with a plausible SPE line at
S/N > 5 and SPE probability > 0.9 (n = 42): agreement 14–19 %, 21–29 % at
S/N > 10 (n = 14).

**Outlier pixels drove the largest statistics.**  Agreement *fell* with my
statistic: 1.9 % in the top bin (> 200).  Those spectra had a median of 4
pixels beyond 5 robust-σ (0 elsewhere): single-pixel spikes at +20 σ, deep
dips, unmasked edges.  38 % of spectra have at least one such pixel.  A
Gaussian matched filter gives Δχ² ~ 400 to any template that lands on a
+20 σ pixel.  `spectra/outliers.py` rejects narrow runs — but the first version
rejected by *width* and removed real lines at S/N 10–20 (the [O III] S/N 44
object's truth statistic went 534+320 → 76).  Rewritten as a *shape* test: a
spike's neighbours are at the baseline, a line's at ≥ 0.6 of its peak.
Regression tests inject lines from S/N 100 down to 5.  With the fix, that
object is identified correctly at statistic 1024.

**Continuum flexibility made it worse**, not better (truth/winner ratio
0.68 → 0.27 from 12 to 50 knots), so the residual failures are not continuum
structure.

**Plotting the seven confident-but-wrong cases** (`plots/confident_wrong.png`)
settled where the remaining disagreement lives:

- in three, nothing is visible at SPE's claimed line (Pa-γ S/N 7, Hβ S/N 5,
  [O II] S/N 19) — SPE is wrong;
- in one, my scan finds Hβ + [O III] 4959 + [O III] 5007 as three clean peaks
  with the right ratios at z = 1.950 while SPE's [S II] sits on nothing — mine
  looks right;
- in one, SPE and I put lines on the same feature with different identifications;
- one has a dominant asymmetric 4×10⁻¹⁷ feature ~100 Å wide — a broad-line
  candidate, not a z = 0.5 galaxy — worth a dither-level look;
- one revealed a bug: the winning template put Pa-γ on a spike in the last
  three pixels before 18500 Å.  Lines within 4 pixels of either end of the
  covered range are no longer tested (`edge_margin_pixels`).

**Conclusion for the truth set.**  SPE agreement cannot be pushed much further
because SPE itself is wrong for a large share of the S/N 5–10 objects that
dominate the strong-line sample.  The comparison has done its job — it found
six real defects — and is now limited by its reference.  Validation moves to
DESI redshifts (44 667 EDF-N galaxies, Δχ²_DESI > 25), which is what the plan
always specified for M4.

Tests: **274 offline**.

---

## 2026-09-17 — session 7: DESI as truth

### Sample

35 810 of the 36 149 DESI galaxies in EDF-N with Δχ²_DESI > 25 and
H ≲ 22.5 have a Q1 spectrum.  Twelve SIR files richest in matches were pulled
(`~/data/euclid/q1/SIR/`, now 32 files, 19 tiles) giving 503 DESI galaxies
with certain redshifts.  Blind recovery against `desi_z`: **1.4 %**.

That number is right and uninformative.  **481 of the 503 are at z < 0.45.**
DESI's Euclid overlap is dominated by bright BGS galaxies, and a z < 0.45 galaxy
has no strong emission line in the red grism — Pa-β is ~5 % of Hα.  The sample
tests the continuum, not line identification.  Its 77 confident-but-wrong
identifications are bright (median S/N 58 per pixel) at z = 0.04–0.26, where
the H-band stellar features — Mg I 1.50 µm, the CO bandheads at 1.56–1.62 µm,
Si I 1.59, Al I 1.675 µm — all fall inside 12500–18500 Å.  The multi-line
templates are fitting the bumps *between* stellar absorption bands.  The
stellar-population continuum the brief deferred "until residuals demand it" is
now demanded, for the bright low-redshift population.

### The right test: the Hα window

9 091 DESI galaxies at 0.9 < z < 1.8 have Q1 spectra (5 788 with Δχ²_DESI > 100).
They are spread thin — about ten per file — so twelve more files gave 114.
Median continuum S/N 3.0: faint ELGs.

Blind agreement: **14 %** overall.  The honest denominator is whether Euclid
detects Hα at all.  A single Hα matched filter placed *at the DESI redshift*:

| Hα Δχ² at DESI z | fraction of sample | blind agreement among them |
|---|---:|---:|
| ≤ 9 (not detectable) | 32 % | 3 % |
| > 9 | 68 % | 19 % |
| > 25 | 50 % | **25 %** |
| > 50 | 28 % | **38 %** |

Where there is nothing to identify, agreement is ~0 as it must be.  Where Hα is
clearly present, the blind scan identifies the redshift 25–38 % of the time.
The residual failures are the single-line ambiguity — Hα against [O III] and
Pa-β with no second line above noise — which the PHZ/SPE hypotheses of the
production configuration exist to break.  That test is running.

### What the experiment established, in one place

- SPE line detections at S/N 5–10 are frequently absent from the data; 17 % of
  SPE lines at S/N > 5 have impossible widths.  Not a usable truth.
- DESI is a clean truth but its Euclid overlap is mostly low-z bright galaxies
  with no grism emission lines; the Hα-window subsample is the line test.
- Six pipeline defects found and fixed (ranking, list order, local refinement,
  templates, outlier shape test, edge margin); two proposed fixes measured to be
  harmful and left off by default (per-component penalty, more knots).
- Bright low-z galaxies need a stellar continuum model.
- One object (2684915737657679255) shows a dominant asymmetric ~100 Å feature
  that is a broad-line candidate mislabelled by SPE as a z = 0.5 galaxy; first
  thing to look at when dither-level fitting exists.

### Closing the single-line ambiguity: the PHZ prior

On the same 114 Hα-window DESI galaxies, the photometric redshift is good —
|Δz|/(1+z) median 0.026, 87 % within 0.1 — and **Euclid's own SPE redshift
agrees with DESI for only 23 % of them** (44 % at SPE probability > 0.99).

Adding SPE/PHZ as *hypotheses* alone lifted Hα-detectable agreement 25 → 30 %:
a hypothesis only helps if it also wins.  PHZ now enters the *ranking* as a
soft prior, −2 ln of a Gaussian (σ = 0.05 in dz/(1+z)) / uniform mixture with a
13 % outlier component.  The mixture caps the penalty near 11 in Δχ² units —
the honest weight of a photometric redshift: it settles moderate ambiguities and
cannot overturn a strong data preference.  A first version with an arbitrary cap
of 40 was replaced when a synthetic test showed the cap doing the arguing.

That test also exposed a template-coverage gap: every Hα template carried
[N II] ≥ 0.25 Hα, so a metal-poor ELG's Hα was better explained by a
single-line Pa-β identification.  `halpha_elg` ([N II]/Hα 0.07) added.

And a precision bug: `iterrows` upcasts a mixed row to float64, and a 19-digit
MER object id does not survive `float → int` (2708573889636910920 came back as
a different integer).  Lookups now key on the int64 index; regression test added.

**Result** (`plots/desi_redshift_recovery.png`):

| configuration | Hα Δχ² > 25 (n = 57) | Hα Δχ² > 50 (n = 32) | Hα not detectable (n = 36) |
|---|---:|---:|---:|
| blind scan + templates | 26 % | 41 % | 3 % |
| + PHZ prior | 53 % | 62 % | 6 % |
| + PHZ prior + SPE/PHZ hypotheses | **60 %** | **69 %** | 3 % |
| *Euclid SPE itself vs DESI* | *23 %* | | |

Agreements sit on the 1:1 line at |Δv| ≈ 100 km/s; the transition to
agreement happens at Hα Δχ² ≈ 20.  Where Hα is not detectable the pipeline
agrees 3 % of the time — as it must, there is nothing to identify.

This is the production identification configuration (`rank_by="template"`,
`phz_prior_sigma=0.05`, catalogue hypotheses on).  The pure-blind statistic is
still computed and recorded on every row, and the winner's origin says whether
the prior decided it.

Caveats stated plainly: n = 32–57; one field; ELGs only; the prior's
parameters were read off this same DESI sample (σ and outlier fraction), so
the numbers are optimistic by an amount a held-out sample will have to measure.

Tests: **277 offline**.

---

## 2026-09-17 — session 8: holdout, and the first injection/recovery

### Holdout

Fourteen fresh SIR files, none used to set the prior, holding 112 DESI galaxies
at 0.9 < z < 1.8; prior parameters frozen at the defaults (σ = 0.05, outlier
fraction 0.13).  Production configuration (blind scan + templates + PHZ prior +
catalogue hypotheses):

| | Hα Δχ² > 25 | Hα Δχ² > 50 | Hα not detectable |
|---|---:|---:|---:|
| training sample (n = 57 / 32 / 36) | 60 % | 69 % | 3 % |
| **holdout (n = 61 / 33 / 22)** | **61 %** | **76 %** | 5 % |
| Euclid SPE vs DESI, same holdout | 25 % | | |

The training numbers were not optimistic.  Cache now 46 files, 33 tiles.

### Injection/recovery framework (`validation/injections.py`)

Broad lines are injected into *real* spectra at redshifts already recovered
against DESI, using each object's own LSF and grid; the M0/M1 measurement runs
at that redshift; recovered flux and Δχ² are recorded against the injection
with LSF, S/N, usable fraction, noise inflation and continuum orthogonality.
Two nulls on the same spectra with nothing injected — M1 at displaced
redshifts, and a broad component forced onto [N II] — give the false-positive
distribution from which thresholds are read.

**First run found a model degeneracy.**  38 targets, fluxes 10⁻¹⁷–10⁻¹⁵,
widths 700/1500/3000 km s⁻¹.  Nulls were tight (p99 of effective Δχ² 1.3 /
4.9 / 2.5 for on-redshift / off-redshift / forbidden), but completeness at
10⁻¹⁵ erg s⁻¹ cm⁻² and σ = 700 km s⁻¹ was **zero**.  Cause: the shared
non-parametric NLR profile spanned ±1000 km s⁻¹, so a 700 km s⁻¹ line is a
perfectly good *narrow* profile and M0 absorbed it — the profile's fitted width
rose to 450–650 km s⁻¹ on injection.  Capping the profile at ±400 km s⁻¹
(step 200) keeps it LSF-limited at ~265 km s⁻¹.  That cap is now the model's
definition of "narrow": anything the profile can represent is narrow, anything
wider must go to the broad component.  Genuinely broad NLRs (σ ≳ 400 km s⁻¹)
will leak into the broad branch at Euclid's resolution; the recorded
`broad_resolution_ratio` is the axis on which that has to be reported.

The remaining smallness of Δχ² is the data, not the model: for a faint ELG
(continuum S/N ≈ 3, ~3×10⁻¹⁸ per pixel) a 10⁻¹⁵ line at 3000 km s⁻¹ is spread
over ~11 pixels at S/N ≈ 1 each, Δχ² ≈ 11.  Euclid's broad-line flux limit for
such hosts is intrinsically several × 10⁻¹⁶, roughly √(σ_broad/σ_narrow) times
the narrow-line limit.

Re-run with the cap, at 12 and 6 continuum knots, appended below.

### Injection/recovery with the profile capped — the first selection function

38 DESI galaxies at 0.9 < z < 1.8 with correctly identified Hα; broad Hα
injected at 7 fluxes (3×10⁻¹⁷ – 10⁻¹⁵ erg s⁻¹ cm⁻²) × 3 widths; 483 null
trials with nothing injected (on-redshift, off-redshift, broad-on-[N II]).
Figure `plots/completeness_desi_halpha.png`.

**Null.**  The three constructions agree: p99 of effective Δχ² = 2.9 / 4.1 / 3.4
(on / off / forbidden) at 12 knots.  Thresholds read from the pooled null:
**1.4 at 5 % FPR, 3.4 at 1 %**.  These are single-hypothesis thresholds — the
redshift is fixed — and are *not* the Stage-1 scan thresholds, where ~1000
hypotheses are tried per object and the null p95 was ~21.

**Completeness (12 knots, FPR 5 %):**

| injected flux | σ = 700 km/s | 1500 | 3000 |
|---|---:|---:|---:|
| 1.8×10⁻¹⁶ | 0 % | 21 % | 16 % |
| 3.2×10⁻¹⁶ | 5 % | 37 % | 29 % |
| 5.6×10⁻¹⁶ | 29 % | 71 % | 47 % |
| 1.0×10⁻¹⁵ | 55 % | 89 % | 79 % |

50 % completeness: **~9×10⁻¹⁶ at 700 km/s, ~4×10⁻¹⁶ at 1500, ~6×10⁻¹⁶ at
3000**.  The width dependence is the two degeneracies made quantitative: 700
km/s is only ~1.7× the LSF for these sources and competes with the narrow
profile; 3000 km/s competes with the continuum (orthogonality 0.47).  1500 km/s
is the sweet spot of this grism at 12 knots.

**Flux bias.**  Unbiased (recovered/injected 0.9–1.0) above 5×10⁻¹⁶.  Below
that, detections are strongly biased high — 2–20× at 10⁻¹⁶ and below — the
Eddington bias of selecting upward fluctuations, now measured rather than
assumed.  Any flux quoted for a marginal candidate must carry this.

**Continuum stiffness.**  At 6 knots the null widens (p99 6–11: continuum
mismatch appears as signal) and the 1 %-FPR completeness falls; 12 knots stays
the default.

Caveats: 38 hosts, all faint ELGs (continuum S/N ≈ 3, LSF 12–20 Å), one
redshift window, one field.  The selection function for bright or extended
hosts, other windows and other fields is not yet measured.  These numbers are
for the *measurement* step at a known redshift; the end-to-end completeness
also carries the identification efficiency (61–76 % on the holdout where Hα is
detectable).

Tests: **285 offline**.  Cache: 46 files, 33 tiles, ~1.6 GB.

### Next

1. Widen the injection sample: more hosts, brighter hosts, the Hβ window,
   EDF-S/F; stratify completeness by LSF and S/N.
2. Stage-1 scan null on the same footing (scan over hypotheses, nothing
   injected) so the Stage-1 threshold is calibrated the same way.
3. M5 dither-level fitting; first target 2684915737657679255.
4. Stellar continuum for the bright low-z population.

---

## 2026-09-17 — session 9: reliability, and what "confident" was measuring

### Redrock-style reliability, first attempt

Adopting DESI's practice: a `zwarn` bitmask (`fit/quality.py`) and purity
against the best-minus-runner-up statistic, read off the DESI comparisons
(218 objects, training + holdout, production configuration).  The result was
the opposite of Redrock's: **purity peaked at ~50 % around a margin of 10 and
fell to 24 % above 30.**  The most confident identifications were the worst.

### What confident-and-wrong was

Decomposition ruled out the broad component (median 27 % of the statistic).
The plot (`plots/confident_wrong_desi.png`) showed strong real features
(Δχ² up to 1900) at neither DESI's Hα nor any catalogue line at DESI's
redshift — in one object DESI's Hα is itself detected at Δχ² = 730 and loses.
The per-dither spectra settled it (`plots/confident_wrong_dithers.png`):

| object | winner's line | excess per dither (σ) |
|---|---|---|
| 2697190336649521081 | "Hβ" 14781 Å | 0.7, **14.5**, –, – |
| 2688846847658475432 | "[O III]" 14673 Å | **29.1**, 4.1, 2.7 |
| 2681252710672318871 | "He I" 17067 Å | **8.8**, 0.7, 0.9, 1.6 |

A neighbour's emission line on one grism orientation.  The combined spectrum
cannot reveal it, and it produces the *strongest* features — hence confidence
anticorrelating with correctness.  Dither coherence is therefore required to
trust a redshift, not only an AGN.

Also found and fixed on the way: `pixel_edges` on the non-contiguous kept grid
gave each masked gap's width to its neighbours, so line columns spanned gaps as
straight lines (a Δχ² of 820 was built on one).  Edges now come from the full
grid.

### Stage-0 signal gate (`spectra/features.py`)

Two redshift-agnostic measurements on every spectrum, before any line list:
the strongest single-line matched-filter statistic at any pixel, and its
excess in each dither.  Gate: feature Δχ² > 25 and > 3σ in ≥ 2 dithers.
Quality bits `NO_FEATURE`, `INCOHERENT_DITHERS`.

| DESI sample (n = 218) | n | agreement |
|---|---:|---:|
| all | 218 | 36 % |
| feature, incoherent | 106 | 22 % |
| **feature, coherent** | **77** | **64 %** |
| no feature | 35 | 17 % |

On the gated sample purity is monotonic in the margin: 71 % (> 5), 74 %
(> 10), 76 % (> 15, n = 29).  Confident-wrong objects were 31 % coherent
(median 1 of 3 dithers), confident-right 100 %.

Cost: only 35 % of these faint ELGs pass — individual dithers are ~2× noisier
than the co-add, so a 3σ per-dither test is stringent.  That is a completeness
cost, recorded, not a bias by host property.  The threshold is a knob to be
set by the selection function, not by taste.

Tests: **306 offline**.

---

## 2026-09-17 — session 10: the signal gate's trade-off, measured

Per-dither excesses stored once for the 218 DESI comparison objects
(`outputs/desi_dither_excess.parquet`), then every gate setting swept
(`validation/gates.py`; figure `plots/gate_tradeoff.png`; table
`outputs/gate_tradeoff.parquet`).  Completeness is measured on the 118 objects
whose Hα DESI-redshift is detectable at Δχ² > 25 — the gate's cost on objects
that genuinely have signal; purity is agreement with DESI among passers.

| feature > 25, per-dither σ | ≥ 1 dither | ≥ 2 dithers | ≥ 3 dithers |
|---|---|---|---|
| 1.0 | 39 % / 98 % | 42 % / 94 % | 50 % / 84 % |
| 2.0 | 41 % / 97 % | 54 % / 80 % | 64 % / 62 % |
| **2.5** | 43 % / 91 % | 61 % / 73 % | **72 % / 54 %** |
| 3.0 | 45 % / 83 % | 64 % / 56 % | 77 % / 38 % |
| 4.0 | 47 % / 69 % | 72 % / 36 % | 78 % / 22 % |

(purity / completeness.)  Two facts from the front:

- **Requiring three dithers dominates requiring two** at every completeness:
  a neighbour's line on one orientation survives a two-dither test whenever a
  second dither has a noise bump.
- **Tightening removes wrong identifications far faster than right ones**:
  across the sweep, wrong passers fall 97 → 13 while correct passers fall
  70 → 33.  That is what a gate against contamination should look like.

Default set at the knee: **feature > 25, 2.5σ in ≥ 3 dithers** — 72 % purity,
54 % completeness on Hα-detectable objects, 63 % of correct identifications
kept.  Both numbers are recorded on every row (`feature_*` columns and the
`INCOHERENT_DITHERS` bit); the gate is a quality flag, not a deletion, so any
downstream analysis can move along the front.

Cost to carry into the selection function: for faint ELGs (continuum S/N ≈ 3),
half of the objects with a real Hα fail the gate because individual dithers are
~2× noisier than the co-add.  The proper cure is the dither-level *fit* (M5),
which uses all the flux rather than a per-dither peak test.

Tests: **311 offline**.

---

## 2026-09-17 — session 11: towards a general-purpose redshift engine

### Why

The user's requirement is redshifts for *any* Euclid spectrum with features,
including z < 0.9, where the red grism carries no optical emission lines and the
information is in the stellar continuum (1.6 µm bump, CO bandheads, Mg I/Al I,
Paschen absorption) plus weak Pa-β/Pa-γ/He I/[S III] emission.  Our identifier
was emission-line-only with a nuisance spline continuum; it cannot do this, and
neither, in practice, does Euclid:

**Baseline on 500 bright low-z DESI galaxies** (z < 0.9, median z 0.18,
median continuum S/N 41 per pixel):

| method | agreement with DESI |
|---|---:|
| Euclid SPE, all (97 % have a solution) | **16 %** |
| Euclid SPE, `spe_z_prob` > 0.99 (n = 253) | 25 % |
| PHZ mode, within Δz/(1+z) < 0.05 | 62 % |
| our emission-line scan | 1.4 % |

SPE's upper quartile lands at z = 1.61 for galaxies DESI puts at z ≈ 0.3: it
manufactures an Hα.  Bright continua with real stellar features and a 16 %
spectroscopic success rate is the opportunity.

### What the community does, and what we take from it

Euclid SPE (AMAZED): BC03 continua + 14 empirical line-ratio templates, three
object classes, redshift PDF and per-class evidence.  DESI Redrock: PCA or
archetype templates per class, χ² on a redshift grid, best-minus-runner-up Δχ²
as reliability, `ZWARN` bits.  QuasarNET: learned per-line detection for QSOs.
Our engine already has the Redrock *structure* (joint linear fit at each z,
class comparison, Δχ² margin, prior, quality bits).  What it lacks is
**templates that reach 1.25–1.85 µm**.

Redrock's own templates were checked directly (files downloaded and headers
read): GALAXY 1228–11000 Å rest, QSO-LOZ 1329–9634, QSO-HIZ 444–4499, STAR
3000–11000.  They cover the Euclid window fully only for z ≥ 0.68 (galaxies),
≥ 0.92 (QSO) and never for stars — built for DESI's 3600–9800 Å.  The
algorithm transfers; the templates do not.

### Libraries obtained (`~/data/euclid/templates/`, provenance in its README)

- **XSL SSP models** (Verro et al. 2022): 390 SSPs, 350–2475 nm, R ≈ 30 000,
  log t = 7.7–10.2, [M/H] −2.2 to +0.2, Kroupa, PARSEC/COLIBRI.  Empirical
  stellar spectra, so the H-band features are real (`plots/xsl_ssp_nir.png`).
  **Telluric gaps** at rest 1.35–1.425 µm and > 1.80 µm (fractional roughness
  9–25 % vs 1–2 % elsewhere) must be masked and bridged; Euclid sees those
  regions cleanly.
- **XSL DR3 stellar spectra** (830 spectra, 683 stars, 350–2480 nm) — for the
  STAR class.  Downloading (772 MB).
- **Glikman et al. 2006 quasar composite**, 0.58–3.5 µm — for the QSO class at
  low redshift; the rest-UV/optical composite for z > 1.5 still to add.
- E-MILES (1680–50000 Å) is behind a password-protected share; XSL supersedes
  it for our purpose.  BC03/BaSeL theoretical NIR spectra are the telluric-free
  complement if bridging the XSL gaps proves insufficient.

### Plan (unchanged from the discussion): 
1. template loader with masking/bridging and Euclid-frame projection;
2. low-dimensional continuum basis per class (PCA/archetypes of the SSPs);
3. class-comparison engine replacing `rank_by="template"` inside the scan;
4. validate on the 218 Hα-window and 500 low-z DESI sets — numbers to beat
   64 % (coherent Hα sample) and 16 % (Euclid SPE at low z);
5. broad-line truth set (DESI QSOs) for the QSO class.

### First continuum-only redshift scan on real data (step 2/3, GALAXY core)

`fit/continuum_redshift.py`: χ² of (5 XSL PCA components + mean + linear
polynomial) at every z on a 300 km/s grid, 0 ≤ z ≤ 1; reliability = Δχ² to the
best minimum more than 3000 km/s away (Redrock's statistic).  Basis: 108 SSPs
with log t ≥ 8.5, [M/H] ≥ −0.5, telluric gaps bridged; explained variance
[0.970 0.015 0.008 0.003 0.001].  Run on the 489 unique low-z DESI galaxies
(`outputs/continuum_redshift_lowz.parquet`, 17.1 min, 2 s/object):

| | agreement within 1000 km/s |
|---|---:|
| all 489 | **30.3 %** (Euclid SPE: 16 %) |
| S/N > 50 / 20–50 / 10–20 per pixel | 59 % / 17 % / 4 % |
| z < 0.15 / 0.15–0.3 / 0.3–0.45 / 0.45–0.9 (n=9) | 54 % / 17 % / 18 % / 0 % |
| purity at Δχ²_runner-up > 10 / 25 / 50 / 100 | 36 / 40 / 46 / 51 % (retained 92 / 78 / 64 / 48 %) |

Median reduced χ² 2.16 (archive variance; the known η² ≈ 2 noise inflation).
Failures scatter (median |Δz| 0.15, only 6 % at the z = 0 grid edge): the
basis is confusing real features, not sliding off the grid.  Purity rising
monotonically with Δχ² is the property the emission-line scan lacked.
`plots/continuum_redshift_lowz.png`.

Two diagnoses (verified):
- The XSL flux inside the telluric gaps is unusable (spikes to ±40× the
  continuum, `plots/xsl_ssp_telluric_gaps.png`), so the bridge is a stiff cubic
  across rest 1.35–1.425 µm — which at z = 0.15–0.3 covers the reddest,
  highest-weight quarter of the Euclid window.  Together with the intrinsically
  feature-poor rest 0.9–1.5 µm window this is the likely cause of the weak
  0.15–0.3 bin.
- The bright low-z galaxies are extended: their `LSF_SIG` is 24–120 Å rather
  than the 13.7 Å point-source median, so the templates must be smoothed
  per object — the LSF is part of the model, not a constant.

### Speed: precomputed template cube (`fit/template_cube.py`)

Every Q1 combined spectrum is on the same grid (531 pixels, 11900–19002 Å,
13.4 Å; checked on 6 tiles), so the redshifted, LSF-smoothed, pixel-integrated
templates are computed once per (template set, LSF bucket of 5 Å) as a
(n_z, 531, n_templates) cube and every object reduces to a batched
pseudo-inverse over its kept pixels.  Smoothing is done once per (template,
LSF) in ln λ — the LSF width in ln λ is redshift-independent — and each z is
two interpolations of the cumulative integral.  `Template.bridged()` is
memoised.  Cube build 0.3 s (6 templates) to 1.3 s (18); per-object scan of
694 redshifts ≈ 10 ms.  Checked against the direct scan: χ² agrees to
1×10⁻⁵ relative (`tests/unit/test_template_cube.py`).  Nonnegative
(archetype/NNLS) mode and the PHZ mixture prior are options of the same scan.
`validation/continuum_experiments.py` runs named variants on the low-z sample
and is the reproducible form of what was previously an inline script.

Pilot on 60 objects: PCA-5 + linear 38 %, archetypes (18 SSPs, NNLS) 52 %.
Full 8-variant sweep on 489 objects running.

### Sweep 1 (8 variants, 489 objects, 6.5 min total) and the failure diagnosis

| variant | within 1000 km/s | z<0.15 | 0.15–0.3 | 0.3–0.45 | purity Δχ²>25 (kept) |
|---|---:|---:|---:|---:|---:|
| PCA-5 + linear | 29.9 % | 54 | 16 | 18 | 41 % (59 %) |
| PCA-3 / PCA-2 / PCA-8 + linear | 30.5 / 30.9 / 27.8 | | | | |
| PCA-5 + constant / + quadratic | 25.6 / 31.9 | | | | |
| 18 SSP archetypes, NNLS, + linear | **34.2 %** | 59 | 21 | 22 | 43 % (68 %) |
| same incl. young (log t ≥ 7.7) | 32.7 % | | | | |

The basis and nuisance choice moves the answer by a few per cent; the PHZ
prior by ~1 % (its cap of ≈11 is nothing against χ² gaps of hundreds).

**Where the truth sits in χ²** (`validation/continuum_diagnostics.py`,
`outputs/continuum_diagnostics_arch_nnls_p1.parquet`): for the 322 failures,
χ²(z_DESI) − χ²_min has quartiles 45 / 114 / 455; only 7 % are within 10.  The
wrong minima are not noise-level ties.  Median reduced χ² is 2.4 (right) vs
3.0 (wrong) on the archive variance.

**Three failure modes seen in the worst high-S/N objects**
(`plots/continuum_failures/`):
1. *Extended sources and tolerance.*  Object 2693182175649381778 (S/N 256,
   LSF σ = 88 Å) is "wrong" by 1750 km/s: with an 88 Å LSF the redshift from
   continuum shape cannot be better than ~10⁻² and a 1000 km/s tolerance is
   the wrong yardstick.  Bright low-z galaxies are extended: LSF_SIG 20–40 Å
   for 70 % of the sample, 40–120 Å for 18 %.  With the catastrophic-failure
   convention |Δz|/(1+z) < 0.01 the archetype scan agrees for **40.5 %**
   (34.2 % at 1000 km/s); LSF 40–80 Å objects: 56 %.
2. *Dither-incoherent contamination.*  Object 2712466074669129632: the four
   dithers read 14/15/10/17 ×10⁻¹⁷ at 1.59–1.68 µm and 0/27/12/22 at
   1.75–1.78 µm; the combined pixels carry no mask bit.  New
   `spectra/coherence.py` flags pixels whose dithers disagree (per-pixel
   reduced χ² > 5 × the object's median, floor 8; grown by one pixel; pixels
   with < 2 usable dithers).  On this object it removes 49 pixels and takes
   the reduced χ² from 81 to 17 — but the redshift is still wrong (0.119 vs
   0.054): masking the artefacts is necessary, not sufficient.
3. *Structure common to all dithers.*  Object 2693698254649518660 (S/N 115)
   has a 40 % depression over 1.3–1.55 µm present identically in all four
   dithers: either real (then the DESI match is questionable) or a systematic
   the dithers share.  No pixel test can reject it.

**What actually controls success: brightness.**  Cross-matching the DESI
file's Euclid photometry (`outputs/continuum_vs_spe_by_hmag.parquet`,
`plots/continuum_redshift_vs_hmag.png`), agreement at |Δz|/(1+z) < 0.01:

| H (Sérsic) | n | continuum (this work) | Euclid SPE | PHZ (< 0.05) | median S/N/pix |
|---|---:|---:|---:|---:|---:|
| 10–15 | 30 | **97 %** | 40 % | 97 % | 160 |
| 15–16 | 61 | **67 %** | 25 % | 75 % | 83 |
| 16–17 | 164 | **50 %** | 25 % | 67 % | 50 |
| 17–18 | 195 | **27 %** | 16 % | 60 % | 30 |
| 18–22 | 50 | 6 % | 6 % | 54 % | 16 |
| all | 489 | 41 % | 20 % | 66 % | 41 |

Match separation, deblending flag, Gaia match and DESI program are flat in
success; DESI Δχ² > 1000 objects succeed at 57 % vs 23 % below (DESI itself is
surer on the bright ones).  Reading: the red grism's continuum at z < 0.3 is
enough for a redshift at H ≲ 16.5, and the information runs out by H ≈ 18 —
the model-mismatch χ² values are large because, with 18 archetypes and ~700
trial redshifts, the wrong minima are fits to noise and few-per-cent
systematics.  In every bin this beats Euclid SPE by 1.7–2.7×, but for
H > 17 the PHZ is the better estimate and the spectrum should refine, not
replace, it.

Sweep 2 (feature-only spline nuisance, all 108 archetypes, cubic) and sweep 3
(coherence mask, 2–5 % systematic flux-error floor) are running.

### Sweep 2: nuisance flexibility and feature-only mode (489 objects)

Agreement with DESI at |Δz|/(1+z) < 0.01 (`outputs/continuum_variants_2.parquet`):

| variant | all | +PHZ prior | z<0.15 | 0.15–0.3 | 0.3–0.45 | 0.45–0.9 (n=9) |
|---|---:|---:|---:|---:|---:|---:|
| 18 archetypes NNLS + linear | 40.5 % | 41.3 | 63 | 30 | 27 | 11 |
| 18 archetypes NNLS + **cubic** | **43.8 %** | **46.0** | 60 | 35 | 33 | 33 |
| all 108 archetypes NNLS + linear | 44.0 % | 44.2 | 64 | 34 | 31 | 11 |
| feature-only (12-knot spline projected out) | 26.6 % | 32.1 | 54 | 12 | 11 | 11 |
| feature-only, 6 knots | 29.2 % | 32.5 | | | | |
| PCA-5 feature-only | 25.4 % | 30.1 | | | | |

A cubic nuisance polynomial lifts the 0.15–0.45 bins from ~28 % to ~34 %:
part of what looked like model mismatch is broad-band calibration/extraction
shape.  Removing the broad-band shape altogether (feature-only) loses 14
points — the continuum *shape* is most of the information at these S/N.
Adopted: archetypes + NNLS + cubic.

### Joint continuum + emission-line scan (`fit/joint_scan.py`)

At each trial z the design is [continuum archetypes | one fixed-ratio line
template per system (Hα complex, Hβ/[O III], Pa-β/[S III], He I/Pa-γ,
[O II]) | cubic].  Line amplitudes ≥ 0 always; continuum coefficients ≥ 0 in
archetype mode; polynomial free — one NNLS per redshift (sign-split free
columns).  At the best z the continuum-only refit gives `delta_chi2_lines`
and the normal matrix gives each template's amplitude S/N.  Unit tests: a
z = 1.2 Hα complex at S/N 8 continuum is recovered against the Pa-β
alternative (the toy first injected a lone Hα — which the scan correctly
called Pa-β at z = 0.126: a single line at 14442 Å *is* both); a line-free
z = 0.1 continuum gets its redshift with `delta_chi2_lines` < 15.
Cost ≈ 0.7 s per object (line columns rebuilt per z; vectorise later).
Running on the 489 low-z galaxies now.

### Sweep 3: coherence mask and systematic floor (489 objects, |Δz|/(1+z) < 0.01)

| variant | agree | +prior | reduced χ² | purity Δχ²>100 (kept) |
|---|---:|---:|---:|---:|
| archetypes + linear | 40.5 % | 41.3 | 2.68 | 59 % (34 %) |
| + coherence mask (5× median, floor 8) | 40.3 % | 41.5 | 2.43 | 60 % (31 %) |
| + coherence mask, 3× median | 40.3 % | 41.5 | 2.43 | (identical: the floor of 8 binds) |
| archetypes + **cubic** + coherence | **43.4 %** | **45.4** | 2.08 | 65 % (26 %) |
| + 2 % / 5 % systematic floor | 40.1 / 39.9 % | | 1.56 / 0.62 | 19 % (12 %) / 0 % (4 %) |
| coherence + 3 % floor | 39.7 % | | 1.01 | 0 % (6 %) |
| feature-only + coherence | 28.6 % | 34.2 | 1.69 | 67 % (10 %) |

- The dither-coherence mask does not change which redshift wins; it makes the
  fit honest (reduced χ² down 10 %) and is kept as hygiene.  It removes a
  median 58 pixels per object — more than contamination alone; breakdown
  checked below.
- A systematic flux-error floor leaves accuracy unchanged and *removes* the
  reliability information: with 3–5 % floors the Δχ² gaps shrink until
  purity no longer rises with the margin.  Rejected.  Reliability thresholds
  will be set with the measured noise inflation η² instead.
- Cubic nuisance is the only material gain in sweeps 1–3 (+3–4 points).

Adopted configuration for the GALAXY class: 18 XSL archetypes (log t ≥ 8.5,
[M/H] ≥ −0.5), NNLS, cubic additive nuisance, coherence mask, PHZ prior
reported alongside the data-only answer.

### rvspecfit (Koposov) — assessment for the STAR class and for method

rvspecfit — https://github.com/segasai/rvspecfit, ASCL 1907.013
(https://ascl.net/1907.013), method in Koposov et al. 2011, ApJ 736, 146;
the DESI Milky Way Survey RV pipeline (Cooper et al. 2023, ApJ 947, 37,
https://doi.org/10.3847/1538-4357/acb3c0) — fits spectra with *interpolated
synthetic PHOENIX templates* (Husser et al. 2013, A&A 553, A6,
https://doi.org/10.1051/0004-6361/201219058) over Teff, log g, [Fe/H],
[α/Fe], RV.  Pipeline (README, verified): PHOENIX grid → SQLite
(`rvs_read_grid`) → per-instrument templates convolved to the instrument
resolution (`rvs_make_interpol`) → n-D interpolator (`rvs_make_nd`, Delaunay
/ multilinear / NN) → CCF templates (`rvs_make_ccf`); fit = CCF initial
guess, then maximum-likelihood optimisation with a *multiplicative*
polynomial continuum (order 10 default); multi-arm simultaneous fitting.

Relevance here:
1. STAR class: PHOENIX has no telluric gaps (unlike XSL) and stars are point
   sources, so rvspecfit's fixed per-configuration LSF is exactly right for
   them; gives Teff/log g/[Fe/H] plus a stellar χ² to set against our
   GALAXY/QSO χ² — DESI's star/non-star recipe.
2. Method transferred to all classes: multiplicative continuum polynomial
   (implemented in `cube_scan(multiplicative_degree=...)`, bilinear solved by
   alternation; sweep 4 running), CCF-then-refine (we have grid + parabolic
   refinement), interpolation in physical parameters (our NNLS mixture of
   archetypes plays this role for galaxies).
3. Not applicable to galaxies/QSOs directly.
Cost: PHOENIX v2.0 grid download (restricted Teff/log g/[Fe/H] range) and a
one-off NISP configuration build (R ≈ 465, 11 900–19 000 Å).  Awaiting the
go-ahead for the download.

### Joint scan on the low-z sample (`outputs/continuum_variants_joint.parquet`)

| variant | |Δz|/(1+z) < 0.01 | within 1000 km/s |
|---|---:|---:|
| archetypes + cubic (continuum only) | 43.8 % | 39.1 % |
| + line templates (joint) | 43.8 % | 36.6 % |
| + line templates + coherence mask | 43.1 % | 36.4 % |

A wash on this sample: the joint fit fixes 25 objects and breaks 25.  Where
the lines carry evidence (Δχ²_lines > 100, n = 99) joint is 24 % vs 19 %; at
the strict 1000 km/s the non-negative line columns lock onto noise at
slightly shifted z and cost 2.5 points.  Expected: bright z < 0.3 galaxies are
line-poor in the red grism (best line template is Pa-β/[S III] or He I/Pa-γ,
never the Hα complex).  The decisive test is the 218-object Hα-window sample
(z 0.9–1.8; line-only scan: 35.8 % on all 218) — running with z ≤ 2.

### Joint scan on the Hα-window sample (218 DESI galaxies, z 0.9–1.8; `outputs/continuum_variants_halpha.parquet`)

| method | |Δz|/(1+z) < 0.01 |
|---|---:|
| emission-line matched-filter scan (session 10, same 218) | 35.8 % |
| continuum only, 18 archetypes + cubic | 1.8 % |
| joint continuum + lines | 12.4 % |
| joint + coherence mask | 12.8 % |

The joint model is *worse* than lines alone at z ≈ 1: with ~26 free
parameters over 1100 trial redshifts the continuum part produces spurious
Δχ² across z comparable to the Hα evidence of a faint ELG (S/N 2–5 per
pixel), and the fixed 200 km/s width and single template per system are
weaker than the matched filter's width grid and per-system identification.
Redrock works at this depth because its GALAXY continuum has few
components.  Consequence for the design: continuum complexity must scale
with S/N — rich archetype mixtures for bright continua, 2–3 components for
faint ones — or the continuum and line scans stay separate hypotheses
combined by evidence.  Testing 2-, 3- and 6-component joint fits on the same
sample.

### Sweep 4: multiplicative polynomial and dither-scatter variance (489 low-z, |Δz|/(1+z) < 0.01)

| variant | agree | +prior | reduced χ² | purity Δχ²>10 (kept) |
|---|---:|---:|---:|---:|
| archetypes + additive cubic | 43.8 % | 40.9 | 2.38 | 43 % (80 %) |
| archetypes × **multiplicative cubic** (rvspecfit-style) | **45.8 %** | 40.7 | 2.50 | 44 % (83 %) |
| multiplicative cubic + additive linear | 43.8 % | | 2.43 | |
| additive cubic + dither-scatter variance | 40.1 % | | 0.66 | 64 % (41 %) |
| **multiplicative cubic + dither-scatter variance** | 45.2 % | 41.9 | 0.70 | **62 % (47 %)** |

The multiplicative correction is worth +2 points (flux-calibration and
aperture errors are multiplicative).  Rescaling the variance by the local
dither scatter puts χ² on an honest scale (reduced χ² 0.7 — slightly over-
corrected, the 3-dof scatter estimate is noisy) and, with the multiplicative
cubic, keeps the accuracy while making the Δχ² margin *mean* something:
62 % purity at Δχ² > 10 for 47 % of objects, against 44 % / 83 % on the
archive variance.  Adopted for the GALAXY continuum class: archetypes ×
multiplicative cubic, dither-scatter variance.

### Low-dimensional joint fits on the Hα sample (218 objects)

PCA-2 / PCA-3 + linear + lines: 5.0 / 3.2 %; 6 / 3 archetypes NNLS + lines:
15.1 / 13.8 %.  Continuum freedom is not why the joint fit trails the
line-only scan (35.8 %); investigating the prior and the line model itself.

### Joint scan on the Hα sample with the PHZ prior (`outputs/continuum_variants_halpha_prior.parquet`)

| variant | data only | with prior |
|---|---:|---:|
| line-only matched-filter scan (session 10, prior) | | 37.6 % |
| joint, 18 archetypes + cubic | 12.4 % | 17.4 % |
| joint, 6 archetypes + linear | 15.1 % | 18.3 % |
| joint, PCA-2 + linear | 5.0 % | 6.4 % |

The prior (available for 99 % of the sample via `phz_mode_1`) closes only a
fifth of the gap.  Remaining structural difference: the line-only scan
projects out one z-independent spline continuum, so its χ²(z) varies only
through the lines; the joint fit refits the continuum at every trial z and,
at S/N ≈ 3 per pixel, that refit's χ² fluctuates by about the number of
continuum parameters — comparable to a faint ELG's Hα Δχ² (30–60).  Testing
with a single continuum template; if confirmed, the design becomes
S/N-adaptive: z-independent nuisance continuum + lines for faint spectra,
archetype continuum + lines for bright ones, chosen per object by evidence.

### Why the joint fit trails the matched filter at z ≈ 1 — found

One-template continuum: 16 % / 21 % (linear / cubic, with prior) — so it was
not z-dependent continuum *noise*.  Three line-right/joint-wrong objects
examined directly (`outputs/continuum_variants_halpha_arch1.parquet`):
at the joint's winning z ≈ 0.26–0.33 the *same* line is Pa-β and the single
old SSP + linear polynomial fits the continuum 150–480 χ² units better than
the same SSP redshifted to the true z ≈ 1.3–1.5, where it would have to
reproduce a young, dusty rest-optical ELG continuum that the archetype set
(log t ≥ 8.5, no dust) does not contain.  The matched filter is immune
because it projects out a 12-knot spline once: its χ²(z) moves only through
the lines.  Template inadequacy, not noise, and not fixable by fewer
templates.

Fix implemented in `fit/joint_scan.py`: ``spline_nuisance`` projects the
spline basis out of the data and of *every* column (continuum templates and
lines) — χ²(z) then responds only to lines and sharp continuum features;
``cube=None`` gives a pure line scan in the same framework (must reproduce
the matched filter).  Running lines-only, 1-, 18-archetype and PCA-3
variants with the spline on the Hα sample.

### Spline-deprojected joint fits on the Hα sample (`outputs/continuum_variants_halpha_spline.parquet`)

| model (all with 12-knot spline projected out) | data only | with prior |
|---|---:|---:|
| **lines only** (matched filter re-expressed in the joint framework) | 25.7 % | **39.9 %** |
| + 1 archetype | 22.5 % | 34.4 % |
| + 18 archetypes (NNLS) | 17.4 % | 30.3 % |
| + PCA-3 (free sign) | 7.8 % | 12.4 % |
| session-10 matched filter (prior) | | 37.6 % |

The re-expressed line scan matches (slightly exceeds) the old matched
filter — the joint framework is sound.  Every continuum column added at
S/N ≈ 3 per pixel costs accuracy: deprojected templates still fit residual
structure differently at each z.  Conversely, on the bright low-z sample the
continuum is the only information and lines are irrelevant.  Design
decision: the GALAXY class runs *two* models — (A) lines + spline nuisance;
(B) archetypes × multiplicative cubic with lines, dither-scatter variance —
and picks per object by evidence (BIC on the same pixels), with the
continuum S/N as the expected discriminant (Hα sample median S/N 2.7; low-z
sample 16–160).

### Star truth set (`validation/star_truth.py`, `outputs/star_truth.parquet`)

MER Gaia-matched sources with point_like_prob > 0.9 in the 33 cached tiles
that have a NISP spectrum: 8,275 (IRSA times out on multi-tile IN lists —
one tile per query).  H (2″ aperture) quantiles 17.3 / 18.75 / 20.2.  SPE's
own classification of the same objects: 3,015 star, 4,290 galaxy, 308 QSO,
662 unclassified — at H ≳ 18.5 a Gaia match is not a clean star label, so
the STAR-class test will use H < 17.5 with good `gaia_match_quality`.

### Adaptive GALAXY engine, first two runs (`fit/galaxy_engine.py`)

Hα sample (218, z 0.9–1.8): the lines + spline model on the dither-scatter-
rescaled variance gives **48.2 %** with the prior (22.0 % data only) — up
from 39.9 % on the archive variance and 37.6 % for the session-10 matched
filter.  The rescaled variance down-weights the incoherent pixels that
produce spurious line detections.

Low-z sample (489): the BIC rule failed — it chose the spline model for 480
of 489 objects (a 12-knot spline always fits a bright continuum better than
18 archetypes × cubic, and BIC's k ln n penalty is small), collapsing the
agreement to 6 % against 41.5 % for the continuum model on the same run.
Absolute fit quality carries no redshift information.  Replaced by a
*redshift-evidence* rule: compare the two curves' best-minus-runner-up
margins on the same variance, with S/N guards (S/N < 5 → lines; S/N > 20 →
continuum).  Re-running both samples.

### Adaptive GALAXY engine, margin rule (`outputs/galaxy_engine_{halpha,lowz}.parquet`)

| sample | chosen lines / continuum | engine, data only | engine + prior | lines model | continuum model | previous best |
|---|---|---:|---:|---:|---:|---:|
| Hα window (218, z 0.9–1.8) | 212 / 6 | 21.6 % | **47.2 %** | 22.0 / 48.2 % | 8.7 % | 37.6 % (matched filter) |
| low-z (489, z < 0.9) | 83 / 406 | 41.9 % | **46.6 %** | 4.5 % | 41.5 % | 45.8 % (continuum sweep) |

One engine now covers both regimes.  Where it chose the continuum model on
the low-z sample it is right for 49.5 % (52.2 % with prior).  The leak: 83
low-z objects with rescaled S/N in the 5–20 band went to the lines model
(4.8 % right) where the continuum model would have given ~40 %; the S/N
guards were set on the archive-variance scale but applied on the rescaled
one (a factor ~√5 lower).  Threshold trade-off measured next; note this is
tuning on the same two samples — the holdout (112 objects) must confirm.

Threshold trade-off (rescaled-S/N scale): a pure switch "lines below a,
continuum above" gives 41.5 / 41.7 / 41.9 / 40.9 % (low-z) and 20.2 / 21.1 /
22.0 / 22.0 % (Hα) for a = 3 / 5 / 8 / 12 — the choice barely matters
because the 5–20 band is hard for *both* models (continuum model 5–27 %
there, lines 3–7 %).  Set `snr_continuum = 8`; margins decide only in
5 < S/N < 8.  The engine is at the ceiling the two models allow; the next
gains must come from the models (young/dusty archetypes; line-width grid).

### rvspecfit installed (package only)

`pip install rvspecfit` → 0.9.0 in the venv; CLI tools `rvs_read_grid`,
`rvs_make_interpol`, `rvs_make_nd`, `rvs_make_ccf`, `rvs_desi_fit` present.
Template preparation needs the PHOENIX grid on disk (`rvs_read_grid --prefix
… --glob_mask …` reads Teff/log g/[Fe/H]/[α/M] from FITS headers), then
`rvs_make_interpol --setup nisp_red --lambda0 11900 --lambda1 19000
--resol_func "x/32.3" --step 13.4` — the NISP LSF is a fixed 13.7 Å σ
(FWHM 32.3 Å) for point sources, so R = λ/32.3 varies from 370 to 590 across
the grism and `--resol_func` expresses that exactly.  Download of the PHOENIX
subset pending the user's go-ahead.

### Holdout confirmation (`outputs/galaxy_engine_holdout.parquet`)

112-object DESI holdout (106 in the cache; z 0.95–1.43; never used for any
tuning): adaptive engine **47.2 %** with the prior (23.6 % data only) —
identical to the 47.2 % on the tuning sample.  Euclid SPE on the same
objects: 26.8 %.  Session-10 matched filter on this holdout: 61–76 % *where
Hα was detectable*, 35 % overall; the engine's overall figure is the one to
compare with.  No sign of overfitting from the S/N-switch tuning.

State of the GALAXY class at the end of session 11: one engine, two models
chosen by S/N and redshift evidence, 47 % at z ≈ 1 (SPE 27 %) and 47 % at
z < 0.9 (SPE 20 %) against DESI at |Δz|/(1+z) < 0.01, with the PHZ prior;
purity rises with the Δχ² margin on the dither-scatter variance.

### PHOENIX downloaded; rvspecfit configuration; first class-purity run

- PHOENIX ACES R = 10 000 bundles for Z = 0, −0.5, −1, −2 (2.33 GB) in
  `~/data/euclid/templates/phoenix/`; 841–846 spectra each, `AWAV-LOG` grid
  3000–25 000 Å (Δln λ = 10⁻⁵), no gaps.  Loader `load_phoenix_star/
  load_phoenix_library`.  PHOENIX STAR PCA (Z = 0, every 2nd Teff, 376
  spectra): 8 components explain 99.86 %.
- `scripts/rvspecfit_nisp_setup.sh`: wavelength file from the log grid,
  `rvs_read_grid` (prefix needs a trailing slash — it concatenates),
  `rvs_make_interpol --resol_func "x/32.3" --step 6.7` (vacuum), `rvs_make_nd`,
  `rvs_make_ccf`.  The CCF step asserted on parameter normalisation because
  [α/M] ≡ 0 in our grid; re-running with `--parameter_names teff,logg,feh`.
- **numpy was upgraded 1.x → 2.5.3 as a side effect of `pip install
  rvspecfit`** (not intended; the venv is shared).  The only breakage was
  the removed `ndarray.ptp()` (fixed to `np.ptp`); 343 tests pass on 2.5.3.
- Class purity, 3-class engine with the XSL PCA-8 STAR class
  (`outputs/class_experiment_xsl.parquet`, 1,217 objects, 311 s):
  STAR wins for 711/749 Gaia-matched point sources — but also for 63 % of
  DESI low-z galaxies and 73 % of Hα galaxies.  An 8-component free-sign PCA
  + polynomial fits any smooth continuum; compared by raw χ² against NNLS
  galaxy archetypes it is simply the more flexible model.  Classes must be
  equally constrained: STAR re-done as non-negative PHOENIX archetypes
  (dwarf + giant sequences every 500 K, Z = 0 and −1).

### Class purity with equally constrained classes (`outputs/class_experiment_phoenix_arch.parquet`)

STAR = 56 non-negative PHOENIX archetypes (dwarf and giant sequences every
500 K, Z = 0 and −1), GALAXY = 18 NNLS archetypes, QSO = Glikman composite
(free sign), all with a linear nuisance; 1,217 objects, 529 s.

| set (label) | GALAXY | QSO | STAR |
|---|---:|---:|---:|
| DESI low-z galaxies (250) | **66 %** (was 30 % with PCA stars) | 24 % | 10 % |
| DESI Hα galaxies (218, S/N ≈ 3) | 44 % | **47 %** | 9 % |
| point sources, SPE = star (394) | 4 % | 8 % | **88 %** |
| point sources, SPE = galaxy (305) | 10 % | 9 % | 80 % |

Constraining the stellar class removed its false wins.  The remaining
over-flexible class is QSO: one composite with a *free-sign* amplitude plus
polynomial is a smooth-continuum model that wins at S/N ≈ 3 over 1,100 trial
redshifts.  Fixed: QSO amplitude non-negative.  For the 305 Gaia-matched
point sources that SPE calls galaxies, our engine says star for 80 % — the
rvspecfit stellar χ² (running) is the independent vote.

rvspecfit on Euclid (wrapper `external/rvspecfit_star.py`, config
`~/data/euclid/rvspecfit/config.yaml`, ±1500 km/s, 4th-order multiplicative
continuum): 0.2 s per spectrum.  Three bright Gaia point sources: Teff
3200–3500 K, v = −54…−98 ± 11–15 km/s (formal), reduced χ² 7.6–20 on the
archive variance; two DESI galaxies forced to v = +520/+790 km/s with
reduced χ² 8.8 / 2.3 — the stellar fit alone does not reject a galaxy, the
class comparison must.

### rvspecfit as the star vote (`outputs/class_experiment_phoenix_arch_rvs.parquet`, 1,217 objects, 796 s incl. rvspecfit)

DESI rule "star if rvspecfit χ² < best GALAXY/QSO χ²" (same rescaled
variance): fires for **0 % of Hα galaxies and 2 % of low-z galaxies**
(2nd-percentile margin +9: rvspecfit never beats the galaxy model on a real
galaxy) and for 46 % of the Gaia-matched point sources — 51 % of SPE's
"stars", 40 % of SPE's "galaxies".  The SPE label does not separate these
two groups at all: identical Teff (3200–5100 K, median ~4000), identical
|v| (55 km/s), identical H (17.2).  They are one population of K/M dwarfs,
and SPE's star/galaxy split among them is noise.  Our NNLS-archetype STAR
class calls 84 % of them stars (88 % of SPE-stars, 80 % of SPE-galaxies).

Flexibility asymmetry, now in the other direction: rvspecfit fits *one*
interpolated star × 4th-order polynomial; GALAXY is an 18-component NNLS
mixture; on half the stars the mixture wins by a little (median margin 0 for
SPE-stars).  So: rvspecfit rule = high purity (~100 % vs DESI galaxies), ~50 %
completeness; archetype STAR class = 84 % completeness, ~90 % purity vs DESI
galaxies.  rvspecfit's parameters are the science product (Teff, log g,
[Fe/H], v at 0.2 s per spectrum); the class decision should use both votes,
with the margin thresholds calibrated on the point-source and DESI sets.

### Class purity v3: non-negative QSO (`outputs/class_experiment_v3.parquet`, 1,217 objects, 786 s incl. rvspecfit)

| set (label) | GALAXY | QSO | STAR | rvspecfit star rule |
|---|---:|---:|---:|---:|
| DESI low-z galaxies (250) | **75 %** (66 → 75) | 13 % | 12 % | 3 % |
| DESI Hα galaxies (218, S/N ≈ 3) | **57 %** (44 → 57) | 31 % | 11 % | 3 % |
| point sources, SPE = star (394) | 6 % | 2 % | **92 %** | ~50 % |
| point sources, SPE = galaxy (305) | 12 % | 6 % | 82 % | ~40 % |

Class margin > 25 is reached by 49 % of low-z galaxies (110 of 122 → GALAXY)
and 77 % of point sources (544 of 577 → STAR).  Remaining QSO false
positives on faint ELGs (31 %) are the next target: the QSO class uses the
continuum-only scan, whereas the GALAXY class in production will use the
adaptive engine (lines + spline at this S/N) — the class comparison must be
made on a common nuisance model, which is the open design question for the
production classifier.

### Stellar line-of-sight velocities: how faint? (`validation/star_velocity_experiment.py`, `dither_wavelength_offsets.py`, `star_velocity_plot.py`)

3,582 Gaia-matched point sources (H < 19, point-like > 0.99), rvspecfit on the
combined spectrum and on each dither (2,550 s).  Gaia G from the ESA archive;
V = G − (G−V)(BP−RP) with the Gaia DR3 Table 5.9 relation (verified).  The
sample is K/M dwarfs: BP−RP median 1.80, G−V = −0.62, V−H = 1.80.

**Per-dither fits are unusable** — velocities scatter over the full ±1500
km/s range at every magnitude, Teff swings 2,900–8,600 K on the same star,
reduced χ² 10–120 at S/N 65 per pixel.  Cause measured by template-free
cross-correlation of each dither against the combined spectrum (223 stars,
H < 17, 899 dither spectra with formal error < 200 km/s): **the dithers'
wavelength zero points scatter by 110 km/s rms (0.42 NISP pixel) within a
star**, formal error 5 km/s, no pointing- or detector-level pattern (medians
within ±30 km/s).  The combined spectrum averages four such offsets → a
zero-point floor of ~55 km/s, and its LSF is slightly broadened by the
mis-registration.  (Re-registering dithers before stacking is a concrete
improvement for stars and for the M5 dither-level line fit.)

**Precision of the combined-spectrum velocity** (`plots/star_rv_precision.png`,
`outputs/star_velocity_mags.parquet`):

| bin | n | rvspecfit formal σ_v | 1.48 × median |v| (upper bound) | |v| < 300 km/s |
|---|---:|---:|---:|---:|
| H 16–17 | 242 | 40 km/s | 70 | 94 % |
| H 17–18 | 1333 | 74 | 109 | 87 % |
| H 18–19 | 1943 | 168 | 236 | 65 % |
| G 17–18 | 407 | 81 | 150 | 83 % |
| G 18–19 | 1327 | 88 | 130 | 78 % |
| G 19–20 | 1311 | 121 | 161 | 76 % |
| G 20–21 | 474 | 224 | 307 | 58 % |
| V 18–19 | 905 | 97 | 155 | 76 % |
| V 19–20 | 983 | 87 | 125 | 81 % |
| V 20–21 | 1233 | 129 | 169 | 73 % |
| V 21–22 | 243 | 186 | 296 | 59 % |

The upper bound includes the stars' true dispersion (tens of km/s), so the
truth lies between the two columns.  The H sequence is the physical one: the
grism sees NIR flux, and at fixed V a redder (cooler) star is brighter in H
and has stronger H₂O/CO features — hence the non-monotonic V curve (V 17–18
is worse than V 19–20 because those are the hotter, feature-poor stars).  No
stars brighter than H ≈ 16 have Q1 spectra in the cached tiles.

Answer: ~50–80 km/s per star for H ≲ 17 (G ≲ 18.5, V ≲ 20 for K/M dwarfs),
~100–150 km/s at G 19–20 / V 20–21, and 200–300 km/s (a pixel) at G 20–21 /
V 21–22, where 40 % of fits are already off by > 300 km/s.  Below that the
velocity is not measured.  The zero-point floor (~55 km/s) and 0.4-px dither
mis-registration are the systematics to fix before the bright end can improve.

### G and V from Euclid photometry alone (`validation/star_photometry_calibration.py`, `outputs/euclid_to_gaia_v.json`)

For stars too faint for Gaia: G − I_E and V − I_E as cubic polynomials in
I_E − H_E (MER 2-FWHM apertures, AB), trained on 5,043 Gaia-matched point
sources (G 17.4–21.2; V via the Gaia DR3 Table 5.9 relation).  Robust scatter
**σ(G) = 0.029 mag** (n = 4,517 after 3σ clipping), **σ(V) = 0.067 mag**
(n = 4,094); round-trip median |ΔG| = 0.022, |ΔV| = 0.057.  Per colour bin:

| I_E − H | n | G − I_E | V − I_E | rms G | rms V |
|---|---:|---:|---:|---:|---:|
| 0–0.5 | 999 | −0.21 | 0.00 | 0.07 | 0.16 |
| 0.5–1.0 | 945 | −0.08 | +0.32 | 0.06 | 0.16 |
| 1.0–1.5 | 1574 | +0.03 | +0.73 | 0.04 | 0.16 |
| 1.5–2.0 | 750 | +0.10 | +1.10 | 0.12 | 0.29 |
| 2.0–2.5 | 34 | +0.14 | +1.44 | 0.17 | 0.49 |

I_E (VIS, 550–900 nm) sits between G and the NIR, so G ≈ I_E to ±0.2 across
the K/M sequence and the colour term is small; V needs the colour and is
good to ~0.15 mag for I_E − H < 1.5, degrading for the reddest dwarfs where
V is 1–1.5 mag fainter than I_E.  `euclid_g_v(I_E, H)` applies it;
`star_velocity_plot.py` falls back to it for stars without Gaia G.
Empirical G − H_E(AB) for the same stars: −0.2 / +0.4 / +1.1 / +1.5 / +1.8 at
BP−RP 0.5–1 / 1–1.5 / 1.5–2 / 2–2.5 / 2.5–3 (fit G − H = −2.11 + 2.56x − 0.41x²).

DR1 scaling (assumptions stated in the reply of 2026-09-17): 68,300 spectra
deg⁻² in Q1 × ~1,900 deg² → ~1.3 × 10⁸ spectra, ~13.5 TB of SIR files,
~300 CPU-days for the 3-class engine at 0.2 s per object.

### Q1 census and predicted yield beyond SPE (`validation/q1_snr_distribution.py`, `outputs/q1_snr_hist.parquet`, `outputs/success_vs_snr.parquet`)

SPE's `spe_cont_snr` (rank 0) exists for 3,226,128 of the 4,307,177 Q1
spectra (1,081,049 have no SPE solution at all).  On 685 DESI objects it is
19.3× our median S/N per pixel (16–84 %: 15.1–21.1) — an integrated S/N —
so per-pixel S/N = spe_cont_snr / 19.3.  IRSA's Oracle backend has neither
LOG10 nor alias GROUP BY; the histogram is `FLOOR(spe_cont_snr/19.3)` in 20
`cntr` chunks, joined to `spe_class`.

| per-pixel S/N | galaxy | qso | star | fraction |
|---|---:|---:|---:|---:|
| < 1 | 494,399 | 16,574 | 26,878 | 16.7 % |
| 1–2 | 491,720 | 43,677 | 52,810 | 18.2 % |
| 2–3 | 367,245 | 51,680 | 47,860 | 14.5 % |
| 3–5 | 433,342 | 73,272 | 68,021 | 17.8 % |
| 5–10 | 384,823 | 60,647 | 88,154 | 16.5 % |
| 10–20 | 181,147 | 20,008 | 66,704 | 8.3 % |
| 20–50 | 91,764 | 6,251 | 56,102 | 4.8 % |
| > 50 | 46,255 | 902 | 55,893 | 3.2 % |

Success vs S/N (DESI truth, |Δz|/(1+z) < 0.01, engine with prior vs SPE):
bright galaxies S/N 10–20 / 20–50 / > 50: ours 14 / 38 / 70 %, SPE 8 / 18 /
27 %, ours-right-SPE-wrong 12 / 28 / 47 %.  ELGs S/N 1–2 / 2–3 / 3–5 / 5–10 /
10–20: ours 52 / 58 / 41 / 32 / 60 %, SPE 21 / 12 / 18 / 15 / 7 %,
ours-not-SPE 39 / 50 / 23 / 24 / 53 %.

**Predicted Q1 yield of redshifts we get right where SPE does not**
(rates × population; the bright-galaxy rates applied to galaxy-class spectra
with S/N ≥ 10, where the DESI sample is representative):

| S/N | galaxy-class spectra | gain |
|---|---:|---:|
| 10–20 | 181,147 | 22,200 |
| 20–50 | 91,764 | 25,300 |
| > 50 | 46,255 | 21,500 |
| **total, S/N ≥ 10** | 319,166 | **~69,000** (SPE right on ~50,000; both on ~30,000) |

Faint regime: the ELG rates hold only for galaxies that *have* lines; using
Euclid's design density of ~2,500 Hα detections deg⁻² (157,750 in 63.1 deg²)
with the S/N-weighted ours-not-SPE rate of 0.34 gives **~54,000** more —
an estimate resting on the design number, not a measured population.

Stars: SPE publishes no velocities; star-class spectra with S/N ≥ 20
(σ_v ≲ 100 km/s): **112,000**; S/N ≥ 10: 179,000; all: 462,000.
Not counted: the 1.08 M spectra without any SPE solution, and the S/N 5–10
galaxies (385,000) where neither pipeline is validated.

Cross-check by a second route (`validation/q1_yield.py`: the 17,306 cached
spectra classified by MER into extended/point sources per H bin, scaled ×249
to Q1, times the H-binned DESI rates): extended-source gain ~14,000 (H 15–19),
i.e. 5× below the S/N-census figure.  The cached tiles are not a fair sample
of Q1 (chosen for DESI overlap; only ~4,000 Q1-scaled extended sources at
H 16–17, and `point_like_prob < 0.1` drops compact galaxies), so the census
route — population-complete over 3.2 M spectra — is the one to quote, with
the cached-scaling result as the lower bound.  Stars agree between routes:
~107,000 point sources with H < 17 (cached scaling) vs 112,000 star-class
spectra with S/N ≥ 20 (census).

## Session 12 — extragalactic globular clusters / UCDs as a redshift-and-RV test

Idea (user): compact sources bound to a nearby galaxy are unresolved, have the
simplest possible SED (one old single-burst population), and their velocity is
known — the host's.  A test of the STAR/GALAXY class decision and of the
velocity zero point that needs no new truth catalogue.

**Feasibility, from the measured RV curves.** Old population: V − H_AB ≈ 1.6.
Brightest ordinary GCs M_V ≈ −10…−11, UCDs/nuclear clusters −12.5…−14.
At 50–100 Mpc a GC is H_AB ≈ 21–22 — far below the S/N ≳ 3 (H_AB ≲ 20) needed
for any redshift, so the 50 < D < 100 Mpc range the user suggested is out of
reach.  At D ≲ 25 Mpc the brightest GCs reach H_AB ≈ 18.5–19 and UCDs 17–18,
where σ_v = 80–170 km/s — and the hosts sit at 1,000–2,600 km/s, i.e. 4–9 NISP
pixels away from the Galactic-star velocity zero.  Expected yield per
elliptical: ~3 GCs brighter than M_V = −10.5 (GCLF peak −7.5, σ = 1.3,
N_GC ≈ 300), plus UCDs and dwarf-satellite nuclei.

**Hosts** (HyperLEDA via WSDB, `leda.main`, cz < 3000 km/s inside the four Q1
field cones; 20 found): NGC 1527 (E-S0, 16.8 Mpc, 1176 km/s), NGC 1340 (E,
16.9, 1183), NGC 1339 (E, 19.9, 1392), IC 2035 (E-S0, 21.5, 1504), NGC 1398
(SBab, 20.0, 1400), NGC 1366 (S0, 18.7, 1308), NGC 1412 (S0, 25.6, 1790),
NGC 2110 (E-S0 Sy2, 33.0, 2312) …

**Data route.**  IRSA's MER catalogue is unusable for this: a 10′ RA/Dec box
query ran 1,402 s and died, and the same query restricted to four tiles
returned HTTP 502.  Instead: CAOM cone → tile IDs (``CONTAINS(p.pt, CIRCLE(…))``
— the reverse argument order and ``INTERSECTS`` are both rejected; ~6 min per
host), association table → SIR file paths, download, then read **RA/Dec from
each object's META HDU** and select by separation.  S/N is measured from the
spectra themselves, which is what matters anyway.  None of the 46 already-cached
files lies within 0.5° of any host.

**Control (cached data, `102041031`).** The same code path on an arbitrary
position: of 168 objects within 4′, 27 have S/N > 3.  The three brightest
(S/N 59–263) are classified **STAR** by the 3-class engine with rvspecfit
velocities +40, −123, +112 km/s — the Galactic-star null the GC test needs,
and a demonstration that the class engine does not manufacture redshifts for
bright point sources.

**Null distribution measured** (`outputs/control_field_velocities.parquet`):
385 objects in six random Q1 fields, S/N-stratified 3 → >100, fitted exactly
as the GC candidates will be (QSO class off — see below), rvspecfit with
±3000 km/s.  The random-field velocity distribution is sharply peaked at zero
and nearly empty where a 15–25 Mpc host's satellites would be:

| rvspecfit v (km/s) | fraction | | S/N > 15 |
|---|---:|---|---:|
| \|v\| < 500 | 53 % | \|v\| < 300 | 78 % |
| **+500 … +1600** | **2 %** | **900–1600** | **2 %** |
| railed \|v\| > 2500 | 18 % | railed | 7 % |

STAR-class fraction rises 75 % (S/N 3–5) → 100 % (S/N > 100); rvspecfit's
formal velocity error is 1266 / 562 / 483 / 68 / 24 km/s at S/N 3–8 / 8–15 /
15–30 / 30–100 / > 100.  So a UCD at H_AB 17–18 (S/N 30–60) is an individual
detection and a bright GC at H_AB 18.5–19 a 2–4 σ one, against a 2 % false
positive rate.

**The QSO class had to be switched off** for this test: on the same control
field with it enabled, 12 of 23 objects at S/N 3–10 were called QSO (one
composite, free z to 3.3 over 1,100 trials is the most flexible hypothesis),
while at S/N > 20 all were correctly STAR.  With GALAXY (z ≤ 0.05) and STAR
only, the control gives STAR for 75–100 %.

### NGC 1527: data route, and first velocities

IRSA's TAP went down mid-session (HTTP 502 on the association table after 8
retries over 11 min; MER unusable all session).  Replaced by a **TAP-free
route**: list ``s3://nasa-irsa-euclid-q1/q1/SIR/<tile>/`` (1 s, ~112 files per
tile), open each file *remotely* (11 s, headers + META only) to get its
objects' positions, and select by separation.  Tile 102021498 contains
NGC 1527 (4.3′ from its centre); 87 of its files lie within 15′ of the host;
45 files yield **4,369 unique objects within 10′** (2,447 at S/N > 3, 834 at
> 10, 290 at > 30), extracted in 82 s with ten threads.  No download needed.

**The host itself** (S/N 1246, 0.4″ from the LEDA position, LSF σ = 91 Å
because it is hugely extended): our GALAXY engine gives z = 0.00535, i.e.
1604 km/s against the true 1176 — a 428 km/s error, which is 0.1 of the
resolution element (FWHM 4100 km/s) for this object.  The class engine calls
it STAR with a huge margin, and that is *correct physics*: at z < 0.01 a
passive galaxy and a star are the same spectrum at R ≈ 450, differing only by
a 0.4-pixel shift and by internal broadening far below the LSF.  The
GC/UCD test is therefore a velocity test, not a classification test.

**First measurement** (146 point-like, LSF σ < 16 Å, S/N > 30 sources within
10′; formal-error cut < 200 km/s):

| | NGC 1527 field | random control |
|---|---:|---:|
| good fits | 26 | 46 |
| \|v\| < 300 km/s | 58 % | 87 % |
| **900–1600 km/s** | **1 (3.8 %)** | 0 (0 %) |
| railed at ±3000 | 35 % | 7 % |

No significant excess yet.  The best candidate is object −620942435478631852
at 2.0′ from the centre, S/N 30: **v = 1137 ± 119 km/s against the host's
1176** — but its fitted Teff rails at the grid edge, so it is not yet
trustworthy.  The five-fold excess of railed fits over the control is
contamination: slitless spectra within a few arcmin of a bright galaxy carry
its light and its neighbours'.

**Bug found and fixed:** 101 of the 146 fits were refused with "template
library doesn't cover this wavelength range" — with the ±3000 km/s search the
spectrum shifted by 1 % runs past the template edge at 19163 Å.  Templates
rebuilt over 11500–19500 Å.  (The earlier stellar-RV results used ±1500 km/s
and were unaffected.)

### Verdict on the GC test: no detection, and the reason is measured

Full sample: 409 point-like (LSF σ < 16 Å) sources with S/N > 10 within 10′ of
NGC 1527, all fitted after the template fix; 143 with formal error < 200 km/s.
Matched control: 395 objects from six random Q1 fields, identical cuts, 208
good fits.

| | NGC 1527 field | matched control | Fisher p |
|---|---:|---:|---:|
| \|v\| < 300 km/s (Galactic stars) | 61 % | 85 % | |
| **900–1600 km/s (host window)** | **2 (1.4 %)** | **3 (1.4 %)** | **1.00** |
| railed velocity \|v\| > 2900 | **19 %** | 3 % | |
| railed Teff (< 2400 K) | 9.8 % | 0.5 % | |
| window **and** railed Teff | 2 | 0 | 0.165 |

**No excess at the host velocity.**  The two objects in the window
(1152 ± 132 km/s at 2.0′, 1343 ± 63 km/s at 8.5′) look at first like
integrated old populations — a single-star fitter should rail to its coolest
template on a globular cluster — but their spectra
(`plots/ngc1527_candidates.png`) are visibly corrupt: candidate A has
order-overlap jumps at 16800/17000/17500 Å and a spike at 17250 Å, candidate B
is flat with a 19000 Å edge artefact.  They are contamination, not clusters.

The 19 % vs 3 % railing rate is the quantitative statement of the problem:
**within 10′ of a bright galaxy, a fifth of slitless spectra are corrupted by
its light and its neighbours' orders.**  Combined with the expected yield —
~3 GCs per elliptical brighter than M_V = −10.5, landing at H_AB ≈ 18.5–19
where σ_v = 200–500 km/s — the test cannot succeed at Q1 depth even for the
nearest suitable hosts.  It would need either a much closer host (D ≲ 5 Mpc,
none in the Q1 fields) or UCD-luminosity targets, of which a 17 Mpc S0 has
one or two.

**What the exercise did deliver** (all new, all measured):
1. **Velocity accuracy on an extended source**: NGC 1527 itself, S/N 1246,
   comes out at 1604 km/s against 1176 — a **428 km/s** error at LSF σ = 91 Å.
   Extended-source velocity zero points are good to ~0.1 resolution element,
   no better; relevant directly to AGN line centroids in resolved hosts.
2. **Contamination near bright/extended hosts**: 19 % of spectra unusable
   within 10′, against a 3 % field rate — a systematic the dwarf-AGN search
   must carry, since dwarfs are often satellites of bright galaxies.
3. **A TAP-free data route** (S3 listing + remote META reads) that works when
   IRSA's catalogue service is down, as it was for most of this session.
4. **Confirmation of the stellar RV work**: 85 % of random-field point sources
   at S/N > 10 land within ±300 km/s, exactly as the star-velocity curves
   predict.
5. Physical point worth remembering: at z < 0.01 a passive galaxy and a star
   are the same spectrum at R ≈ 450 — the class engine calling NGC 1527 a STAR
   is correct behaviour, not a bug.

The other seven hosts were never resolved: IRSA's CAOM endpoint returned
HTTP 502/504 for every cone between 18:30 and 19:13.  No point retrying — all
are at 17–33 Mpc, i.e. further than NGC 1527, so they carry fewer usable
clusters at the same contamination level.  If the test is ever revisited it
needs a host at D ≲ 5 Mpc (none in the Q1 fields) or DR1's wider footprint.

**Contamination is radial, and confined to the optical body.**  Splitting the
143 good fits by separation from NGC 1527 (D25 = 4.5′):

| separation | n | railed ±3000 | \|v\| < 300 |
|---|---:|---:|---:|
| 0–3′ | 44 | **52 %** | 18 % |
| 3–6′ | 38 | 8 % | 79 % |
| 6–10′ | 61 | **2 %** | 80 % |
| control, same field (EDF-S, b = −46°) | 28 | 0 % | 96 % |

The 19 % field-average railing comes entirely from inside the galaxy.  Beyond
3′ Euclid's spectra are as usable as in a blank field — so proximity to a
bright host is not by itself disqualifying for the dwarf-AGN search; being
*inside* the host's optical extent is.  It also condemns candidate A (2.0′,
52 % railing zone) while candidate B (8.5′) fails on its own edge artefact.
Caveat on the control: 178 of its 208 objects are an EDF-N field at b = +31°,
against b = −47° for NGC 1527; the same-region control (tile 102021017,
b = −46°) gives 96 % within ±300 km/s from 28 objects.
`plots/ngc1527_velocity_by_radius.png`.

### Correction: the formal-error cut admitted railed fits

rvspecfit reports a *small* formal error for velocities parked at the search
boundary — 23 of the 54 railed inner-region fits passed σ_v < 200 km/s, one
with σ_v = 0.0 km/s at v = −3000.  Honest non-railed fits have a median error
of 820 km/s, so the cut preferentially selected junk.  Re-analysed requiring
|v| < 2900 as well:

| sample | n | \|v\| < 300 | in 900–1600 km/s |
|---|---:|---:|---:|
| NGC 1527, 0–3′ | 21 | 38 % | 1 (4.8 %) |
| NGC 1527, 3–10′ | 95 | **83 %** | 1 (1.1 %) |
| control | 201 | 88 % | 3 (1.5 %) |

The earlier 61 % vs 85 % comparison was distorted by railed junk: outside the
optical body the field is statistically identical to a blank field (83 % vs
88 %).  The non-detection stands and is now on honest numbers (1/95 vs 3/201,
Fisher p = 1.00).

**Why the inner spectra fail is *not* a wavelength-scale error.**  In slitless
spectroscopy wavelength is set by source position — one NISP pixel = 13.4 Å =
259 km/s = 0.30″, so a 1″ centroid error would be 863 km/s.  Tested directly:
velocity versus offset from the galaxy centre gives r = +0.10 (dx) and +0.16
(dy) with a slope of +2 km/s per arcsec, and the railed objects split evenly
between +3000 and −3000 on both sides.  There is no coherent positional shift
— expected, since the four dithers use different grism orientations, so a
positional offset produces different shifts per dither that average out.  The
inner spectra are *incoherently* contaminated: the absorption features that
carry velocity are destroyed, the fit is unconstrained, and it rails.  Nothing
is measured, rather than something measured wrongly.

**Power of the experiment, stated plainly**: ~3 GCs brighter than M_V = −10.5
are expected among the 95 clean-zone objects (3 %), against a 1.5 % control
rate in the same velocity window — ≈ 1.4 expected background events.  Even a
perfect measurement of all three clusters would give Poisson p ≈ 0.2.  The
test could not have succeeded; that is a statement about Q1 depth and GC
luminosity functions, not about the pipeline.

### Resolved (extended) sources in the NGC 1527 field

rvspecfit cannot be used on them — its template library carries the
point-source LSF (R = λ/32.3 Å) while these objects have effective LSF σ =
20–120 Å — so they are measured with our own archetype engine, which builds a
template cube per LSF bucket (`validation/extended_velocities.py`, two passes:
wide 0 < z < 1 at 300 km/s to separate local from background, then a joint
continuum+line scan at 50 km/s for anything below z = 0.03).  282 candidates
(LSF σ ≥ 20 Å, S/N > 10) within 10′ of the host, 146 in the control fields.

**Velocity precision on resolved sources, calibrated against DESI** (228 low-z
galaxies where our redshift is correct): robust σ = **229 km/s at S/N > 50**,
600 km/s at S/N 20–50, 807 km/s below — **unbiased** (median residual
−3 km/s).  So the engine *is* capable of separating 1176 km/s from zero at
5 σ, provided the source has S/N > 50.

**Result** (`plots/ngc1527_extended_velocities.png`): 86 % of resolved sources
are background galaxies (z ≈ 0.1–0.5, indistinguishable from the control
field).  Of the 37 the wide scan places below z = 0.03, 23 are parked at a
grid edge (v = 0 or 9033 km/s) and only 8 carry real redshift evidence
(Δχ² > 9).  Three fall in the 900–1600 km/s window against 0 in the control
(Fisher p = 0.55), but two have Δχ² of 0.43 and 0.00 — no evidence at all —
and the third, a S/N 105 source 1.5′ out with v = 1052 km/s and
Δχ²_lines = 40, has a spectrum with flux steps from 22 to 9 ×10⁻¹⁷ at
13900 Å and back at 16500 Å (`plots/ngc1527_extended_candidate.png`): the
"lines" and the velocity are extraction artefacts.

So the resolved sample gives no detection either, for a different reason than
the point sources: not contamination-induced railing, but that the field
simply contains no bright resolved satellite of NGC 1527 in Q1's spectra.
The useful product is the calibrated resolved-source velocity precision
(229 km/s at S/N > 50, unbiased), which is a number the AGN work needs
directly and which we did not have before.

### Fit gallery, and a decontamination artefact found through it

`validation/fit_gallery.py` renders every measured spectrum with the model we
fitted to it — one panel each carrying S/N, LSF width, winning class, velocity
or redshift, Δχ² and reduced χ², sorted by S/N, twelve to a page.  The model is
re-solved at the best redshift by the new
`fit.template_cube.best_fit_model`, which reproduces the scan's χ² to 2 %
(tested); it exists because the multiplicative continuum polynomial is solved
by alternation and its factor is not part of the stored coefficient vector.

NGC 1527 field, 700 highest-S/N objects (348 unresolved, 235 resolved):
**665 panels on 56 pages**, `plots/gallery_ngc1527/gallery.pdf` plus the first
six pages as PNG and an index parquet.  441 fitted best as GALAXY, 224 as
STAR; **median reduced χ² = 0.70** — the dither-scatter variance puts χ² on an
honest scale — and only 18 (3 %) exceed χ²ᵣ = 20.

Reading the pages turned up a recurring artefact invisible in summary
statistics: a **rectangular flux deficit** between ~13800 and ~16800 Å with
normal flux on both sides.  Quantified (median flux 14300–16300 Å over the
median outside, factor-of-two threshold):

| | fraction with the trough |
|---|---:|
| NGC 1527, 0–3′ | **51 %** (60/117) |
| NGC 1527, 3–6′ | 0 % (0/46) |
| NGC 1527, 6–10′ | 1 % (1/70) |
| blank control fields | **0 %** (0/241) |

It is the SIR decontamination step removing far too much flux over the range
where the bright neighbour's trace crosses the object's own.  It is confined
to exactly the radius where velocity fits rail, so this single failure mode
accounts for the inner-region results reported above — not "contamination" in
the vague sense but an identifiable, diagnosable defect.

`spectra/artefacts.py::continuum_trough` detects it template-free from the
spectrum shape (longest contiguous stretch below half the median level,
bounded by normal flux, ≥ 500 Å wide, not touching an edge), with a new
`ZWarn.DECONTAMINATION_TROUGH` bit.  Validated on 300 gallery objects: flags
20 % overall, 39 % / 0 % / 2 % at 0–3′ / 3–6′ / 6–10′, and flagged objects have
median χ²ᵣ = 2.5 against 0.98 for clean ones.  This is a concrete gain for the
AGN search, where a rectangular deficit across the band would wreck a
broad-line measurement in any host with a bright neighbour.

### A real bug, found by looking at the gallery: the archive LSF over-smooths our templates

The first gallery showed models running straight through obvious absorption
features.  Chased down quantitatively on the twelve brightest clean-zone
objects: the data carry 0.6–1.9 % narrow structure while **our models carried
0.00–0.29 %** — effectively featureless — with individual dips reaching −19 σ.

Diagnosis, in order:
1. *Is the structure real?*  Stacking 236 continuum-normalised spectra in the
   observed frame shows only 1–2 % *common* features (14888, 15022, 13133 Å …
   at 4–8 σ in the median), so the per-object structure is not a shared
   calibration pattern.  Per object, the structure has a correlation length of
   1–3 pixels and repeats between dithers with r = 0.90–1.00, so it is signal,
   not photon noise.
2. *Why is the model smooth?*  Every one of those objects has header
   `LSF_SIG` = 45–73 Å, and we smooth the templates by it — FWHM 110–170 Å,
   which erases all stellar features.
3. *What width do the data want?*  Scanning the template smoothing width:

   | object | header LSF | best LSF | χ²ᵣ (header) | χ²ᵣ (best) |
   |---|---:|---:|---:|---:|
   | −619419693480225144 | 63 Å | 14 Å | 1.31 | 0.76 |
   | −619857270478167031 | 69 Å | 10 Å | 4.95 | 2.84 |
   | −620884294478453080 | 69 Å | 10 Å | 6.27 | 2.94 |
   | −620518401479614289 | 67 Å | 10 Å | 1.35 | 0.69 |

So **`LSF_SIG` is not usable as a template smoothing width.**  Whatever it
measures (it tracks the source's extent, and these objects sit on NGC 1527's
light), the spectra themselves are resolved at close to the instrumental
value.  The grid is clamped at 13.7 Å, NISP's nominal point-source LSF,
because nothing on the sky is sharper than the instrument; fits do keep
improving below that, which is a warning that pixel-scale resampling
residuals exist, not a licence to fit them.

Effect on the same 36 objects: median χ²ᵣ **2.98 → 1.63**, unfitted > 4 σ dips
per object **4 → 2**, objects with no unfitted dip **13 % → 33 %**, median
fitted LSF 14 Å against a header median of 62 Å.  Visually the models now
trace the absorption (`plots/gallery_ngc1527_fitlsf/` versus
`plots/gallery_ngc1527_headerlsf/`).

`fit_lsf` is now an option of both the gallery and `GalaxyEngine`, and the
gallery marks every absorption feature the model misses by more than 4 σ with
a blue triangle, so the failure mode is visible rather than buried in a
summary statistic.  **The DESI low-z revalidation with the fitted LSF is
running — the 46.6 % baseline was obtained with the over-smoothed templates,
so it may be an underestimate of what the engine can do.**

**What the LSF fix does and does not buy** (DESI low-z sample, 489 galaxies,
`outputs/galaxy_engine_lowz_fitlsf.parquet`):

| | header LSF | fitted LSF |
|---|---:|---:|
| agreement with DESI, engine + prior | 46.6 % | 45.4 % |
| velocity residual where both are correct (n = 215) | 280 km/s | **230 km/s** |
| median smoothing width used | 27 Å | 22 Å |

7 objects are fixed and 13 broken — a wash at n = 489.  So the over-smoothing
was a genuine defect of the *model* (featureless templates, χ² doubled) and it
costs **velocity precision**, but it does not change the redshift
identification rate: identification at a 0.01(1+z) tolerance rides on the
broad continuum shape, while the narrow features set the precision.  Sharper
templates can also lock onto pixel-scale residuals at the wrong redshift,
which is presumably why the identification rate does not improve.

The size of the error depends on environment: in blank fields the header is
only mildly inflated (27 Å where the data want 22 Å), but around NGC 1527 it
reads 62 Å where the data want 14 Å.  `LSF_SIG` appears to be inflated by a
bright neighbour's light, which is exactly where it does most damage.

### Velocities by source size and radius (`plots/ngc1527_velocities_resolved_unresolved.png`)

The figure the point-source analysis needed, now covering both populations,
with the resolved half measured by our archetype engine with a **fitted** LSF
(the archive value over-smooths; see above) rather than rvspecfit, whose
template library assumes a point source.

| | 0–3′ | 3–10′ |
|---|---|---|
| **unresolved** (LSF σ < 16 Å) | 21 measured, \|v\|<300 = 38 %, 1 in the host window | 95 measured, \|v\|<300 = 83 %, 1 in the window |
| **resolved** (LSF σ ≥ 20 Å) | 74 fitted → 24 at z<0.03 → **2** with Δχ²>9 | 186 fitted → 15 at z<0.03 → **3** with Δχ²>9 |

Two things the split makes plain.  First, the unresolved inner sample is
wrecked (the stellar peak at zero is gone, 38 % against 83 % outside) while
the outer sample matches a blank field — the decontamination trough again.
Second, **the resolved population cannot test this at all**: 85 % of resolved
sources are background galaxies, and of the 39 the wide scan places locally
only 5 carry real redshift evidence.  Against a control that yields 9 such
objects, none of them near 1176 km/s, there is nothing to detect.

With the fitted LSF the resolved velocities do improve — control \|v\| < 300
rises from 50 % to 69 %, median reduced χ² 0.72 in the host field — but the
sample size, not the precision, is what defeats the measurement.

**Full gallery with the fitted LSF** (`plots/gallery_ngc1527/`, 665 panels, 56
pages, 448 unresolved + 217 resolved, S/N > 10):

| | header LSF | fitted LSF |
|---|---:|---:|
| median reduced χ² | 0.70 | 0.67 |
| unfitted > 4 σ absorption dips per object | median 4 | **median 0** |
| objects with no unfitted dip | 13 % | **74 %** |

What still carries the ✗ flag is broken data rather than a failed model —
flux steps and decontamination troughs.

Caveat on the fitted width: across the whole sample the preferred LSF median
is 30 Å against a header median of 16 Å, the *opposite* direction from the
bright objects (63 → 14 Å).  For faint spectra a broader kernel wins because
a smoother model does not chase noise, so there the width acts as a
regularisation knob, not a resolution measurement.  The defensible statement
is that `LSF_SIG` is unreliable in both directions and the width should be
fitted, with the instrumental floor imposed.

### A 12.8σ "detection" that was an artefact pile-up — and the cut that kills it

The inner-3′ unresolved panel shows a bump at ~700 km/s: **4 of 21 objects in
450–900 km/s against a control rate of 1.5 %** (0.31 expected) — a 12.8×
overdensity, Poisson p = 0.0003, Fisher p = 0.002.  It is entirely spurious:

| object | sep | S/N | v (km/s) | trough | largest pixel jump |
|---|---:|---:|---:|---|---:|
| −620995641479320082 | 2.1′ | 25 | 674 ± 92 | yes | 18× |
| −621229091479302307 | 2.2′ | 15 | 686 ± 60 | no | 79× |
| −621217271479233254 | 1.8′ | 55 | 664 ± 152 | yes | 43× |
| −620947129478634764 | 2.0′ | 24 | 772 ± 192 | yes | 40× |

The tell is that four independent objects agree to 50 km/s while their own
formal errors are 60–192 km/s.  Three share a sharp edge at ~18908 Å and
three have troughs spanning ~14.7–16.2 kÅ: when the fitter aligns template
features to an artefact edge at a *fixed observed* wavelength, every object
carrying that edge inherits the same spurious velocity.  Artefacts do not
scatter — they cluster, and clustering is exactly what a detection looks like.

**The cut works.**  Two template-free shape tests — `continuum_trough` plus a
largest-single-pixel-jump > 10× median criterion:

| sample | n | survive | in 450–900 km/s |
|---|---:|---:|---:|
| NGC 1527 0–3′ | 21 | 5 (24 %) | **4 → 0** |
| NGC 1527 3–10′ | 95 | 94 (99 %) | 3 → 3 |
| control | 201 | 197 (98 %) | 3 → 3 |

Artefact rate 76 % inside 3′ against 1 % outside and 2 % in blank fields.  The
cut removes the false signal and leaves clean data untouched.  Flagged tables:
`outputs/gc_velocities_ngc1527_flagged.parquet`,
`outputs/control_velocities_flagged.parquet`.

**Carry this into the AGN search**: a velocity or redshift pile-up can be
manufactured by shared artefacts at fixed observed wavelengths, and it will
look statistically overwhelming.  Every clustering or line-offset claim must
pass the artefact cut before it is believed.

### Why Q1 object IDs are negative in the southern fields

`object_id` encodes the source position and carries the declination sign.
Checked on all 5,506 cached objects: every negative id has dec < 0 and every
positive id dec > 0, no exceptions — EDF-N (dec +63…+65) gives ids ≈ 2.7×10¹⁸,
EDF-S/F (dec −51…−28) give ≈ −5×10¹⁷…−6×10¹⁷.  NGC 1527 is in EDF-S, hence the
minus signs throughout this session.  Ids must stay int64: float64 silently
corrupts 19-digit values (a bug already hit and fixed in session 10).

## Known quasars in Q1: the QSO class validated against DESI DR1

Truth from WSDB: `desi_dr1.zpix` (zwarn = 0) joined to `desi_dr1.photometry`
for positions.  **6,432 DESI quasars inside the EDF-N cone** (z = 0.05–5.82,
5,191 above z = 0.9); DESI does not cover the southern deep fields.  Matched
to the local SIR cache by position (`validation/qso_truth.py`, positions read
from META HDUs, no catalogue service needed): **150 quasars with Euclid
spectra**, median separation 0.01″, z = 0.17–3.06.

Fitted with the three-class engine — GALAXY archetypes, Glikman composite,
PHOENIX stars, all non-negative, per-object fitted LSF
(`validation/qso_experiment.py`, new `RedshiftEngine.run_at_lsf`).

| | ours | Euclid SPE |
|---|---:|---:|
| redshift right (\|Δz\|/(1+z) < 0.01), all 149 | **44 %** | 30 % |
| z 0–0.9 (n = 32) | **47 %** | **0 %** |
| z 0.9–1.5 (n = 44) | **57 %** | 41 % |
| z 1.5–2.5 (n = 59) | 37 % | 36 % |
| z 2.5–3.5 (n = 14) | 29 % | 43 % |
| S/N > 10 (n = 22) | **77 %** | **9 %** |
| classified QSO | 62 % | 23 % |

SPE labels 99 of 137 of these confirmed quasars **galaxies**, and gets none of
the z < 0.9 ones right — the same low-redshift blind spot seen for galaxies.
Our advantage is largest exactly where the AGN search lives: low redshift and
decent S/N.  SPE is better only at z > 2.5, where the Glikman composite's
blue end (2762 Å rest) starts leaving the grism.

**Failures are line confusion, as expected**: of 83 misses, Hα↔Hβ accounts for
17, Hα→[O III] for 7, Hα→Pa-β for 4; 11 rail at z = 0.

**The margins work.**  Median redshift margin 132 for correct against 7 for
wrong; class margin 222 against 11.  Purity rises monotonically:

| Δχ² cut | kept | ours right | SPE right on the same objects |
|---:|---:|---:|---:|
| 0 | 100 % | 44 % | 30 % |
| 25 | 44 % | 64 % | 42 % |
| 100 | 30 % | 78 % | 53 % |

and requiring the QSO class as well: **Δχ² > 25 and class = QSO gives 81 %
purity for 35 % of the sample; Δχ² > 100 gives 97 % for 24 %.**  That is a
usable AGN selection with a measured selection function — the thing M6 needs.

The trough flag matters here too: 8 % of the quasars are flagged, and their
redshifts are right 25 % of the time against 46 % for the clean ones.

## Session 13 — the broad-line decomposition, updated

The M0/M1 machinery from sessions 4–6 was intact and its 50 unit tests passed,
but it predated the continuum work, the LSF finding and the artefact detector.
Checking it against a *known* broad line exposed a serious defect.

### The nuisance continuum was eating (or inventing) the broad line

Synthetic quasar: power-law continuum plus one broad Hα of known flux at
z = 1.2, S/N 60, fitted with the real machinery.  Recovered flux as a fraction
of truth, and the detection statistic:

| σ (km/s) | spline 8 knots | 12 | 20 | 30 | **power law** |
|---:|---:|---:|---:|---:|---:|
| 1500 | 98 % | 100 % | 75 % | 51 % | **98 %** |
| 3000 | 121 % | 132 % | 69 % | 0 % | **110 %** |
| 6000 | 156 % | 306 % | 0 % | 5482 % | **107 %** |
| 10000 | 0 % | 1964 % | 0 % | 0 % | **100 %** |

| σ (km/s) | Δχ², spline k=12 | Δχ², power law |
|---:|---:|---:|
| 1500 | 52 | **65** |
| 3000 | 49 | **80** |
| 6000 | 19 | **39** |
| 10000 | 7 | **15** |

A B-spline with the knot spacing used for narrow-line work is not stiff on the
scale of a broad line: it either swallows the line or blows the flux up by an
order of magnitude, and **it halves the detection statistic for the widest
lines** — precisely the luminous quasars (FWHM ≳ 12,000 km/s) the search is
for.  `ScreenSettings.n_knots` is 12, so this was the configuration in use.

Fix: `spectra/continuum.py::power_law_block`, a four-column basis
``(λ/λ₀)^index`` that spans quasar continua and *cannot* bend on the scale of
a line; `models/forward.py::continuum_for` makes the block pluggable
(``"spline"``, ``"power_law"``, or a caller-supplied ``(design, penalty,
names)`` triple — which is how stellar archetypes at the fitted redshift will
be passed for host-dominated AGN).  `fit_hypothesis(..., continuum=...)` and
`ScreenSettings.continuum` carry it through.  The default stays ``"spline"``
so narrow-line results are unchanged until the switch is validated on the
redshift samples; broad-line measurement must set ``"power_law"``.
Bias table: `outputs/broad_continuum_bias.parquet`; regression test
`tests/unit/test_continuum_choice.py`.  351 tests pass.

### AGN detection measured on confirmed AGN, and what the continuum fix buys

`validation/agn_measurement.py` runs the M0/M1 decomposition on the 131 DESI
quasars with Euclid spectra, redshift **held at the DESI value** so the
decomposition is judged on its own; per-object fitted LSF; dither-scatter
variance; trough flag recorded; widths scanned 750–9000 km/s over every
permitted line in range, with containment and a new orthogonality guard.

**Orthogonality of a broad Hα column to the continuum basis** (z = 1.2) — the
clearest statement of why the spline was wrong:

| σ (km/s) | FWHM | power law | spline k=12 |
|---:|---:|---:|---:|
| 750 | 1766 | 0.96 | 0.83 |
| 3000 | 7064 | 0.86 | 0.38 |
| 8000 | 18839 | **0.62** | **0.02** |

At FWHM ≈ 19,000 km/s a 12-knot spline reproduces the broad line to 98 %: it
*is* the continuum model.  `broad_orthogonality` now guards the measurement as
it has guarded screening since session 5.

**Detection on real AGN** (same objects, same everything except the continuum):

| | power law | spline k=12 |
|---|---:|---:|
| Δχ² > 25 | **52 (43 %)** | 28 (23 %) |
| Δχ² > 100 | 21 (18 %) | 13 (11 %) |
| S/N < 3 | **32 %** | 6 % |
| S/N 3–10 | **49 %** | 34 % |
| S/N > 10 | **82 %** | 64 % |
| median detected FWHM | 5900–8200 km/s | 4200–5900 km/s |
| lines carrying detections | Hα 33, Hβ 9, He I 5, Pa-β 3, Hγ/Hδ 2 | Hα 17, Hβ 7, He I 3, Pa-β 1 |

**The continuum fix nearly doubles the AGN detection rate on confirmed
quasars** (43 % against 23 %), and the gain is largest at low S/N (32 % vs
6 %) — the faint end where the survey's science lives.  The spline also biases
the recovered width low (median FWHM 4200–5900 against 5900–8200 km/s),
exactly as the synthetic test predicted: it absorbs the broadest wings.

First measured number of its kind for this project: **detection completeness
on real AGN is 43 % overall and 82 % at S/N > 10**, against a false-positive
rate still to be recalibrated in this configuration.  Median broad flux of the
detections is 3.6×10⁻¹⁵ erg s⁻¹ cm⁻² with broad-line S/N ≈ 10, and only 6 % of
widths rail at the grid edge (was 21 % on the coarse grid).

Correction to a claim I made while working: the median broad flux is *not*
"30× too bright for a quasar".  For a z ≈ 1 quasar with F_λ ≈ 2×10⁻¹⁷ and rest
Hα equivalent width ~300 Å the expected line flux is ~10⁻¹⁴, so these values
are if anything modest.

## AGN detection measured on 2,754 confirmed quasars

The 131-object sample was a cache artefact, not a data limit.  Built an index
of **every EDF-N object with a Q1 spectrum** (`validation/edfn_index.py`, MER
tile by tile, 123 tiles, 45 min): **1,683,630 objects**, H_AB median 21.7,
17 % point-like.  Matching DESI DR1 quasars against it:

* 3,151 DESI quasars in the indexed area, **2,755 (87 %) have a Q1 spectrum**
  — not the 36 % I estimated earlier from a crude sky-coverage test;
* z distribution 451 / 680 / 1,229 / 395 in 0–0.9 / 0.9–1.5 / 1.5–2.5 / >2.5;
* they sit in 1,581 SIR files, median **one quasar per file**.

Fetching them: walking to the right HDU over S3 costs 16.7 s per file (threads
gain little — astropy parses thousands of headers under the GIL; processes
gave 8 s), and downloading is 110 MB per file for a few tens of kB of use.
IRSA's per-spectrum API (`api/spectrumdm/convert/...?hdu=N`, the association
table's own ``path``) returns a VOTable with WAVELENGTH/SIGNAL/VAR/MASK/NDITH
in **2.3 s**, so **2,754 spectra arrived in 6.4 minutes** into a 10 MB store
(`validation/qso_extract.py`, `outputs/qso_spectra_edfn.npz`).  Caveat: the
API serves one HDU, so there are no dithers and no per-object variance
rescaling; the measured global inflation η² = 2 is applied instead, which is a
floor rather than a correction.

**Calibrating the orthogonality guard on the data.**  The power-law continuum
cured the spline's flexibility but introduced its mirror image: with four
stiff terms, residual curvature had nowhere to go and the widest Gaussian
absorbed it.

| min orthogonality | detected | widths railed at the scan edge | median FWHM |
|---:|---:|---:|---:|
| 0.5 | 50/105 | 34 % | 11,774 |
| 0.7 | 48/105 | 6 % | 8,242 |
| **0.8** | **45/105** | **0 %** | **8,242** |
| 0.9 | 31/102 | 0 % | 4,239 |

0.8 keeps 90 % of detections and removes railing entirely; it is now the
default and is part of the measurement definition, so completeness must be
quoted with it.

**Result on 2,754 confirmed AGN** (2,379 with a permitted line in range,
272 s):

| | detected (Δχ² > 25) |
|---|---:|
| all | **40 %** |
| z 0–0.9 | 44 % |
| z 0.9–1.5 | **56 %** |
| z 1.5–2.5 | 35 % |
| z > 2.5 | 22 % |
| S/N < 3 | 26 % |
| S/N 3–10 | 48 % |
| S/N > 10 | **68 %** |

Δχ² > 50: 28 %; > 100: 19 %.  Detected median flux 2.0×10⁻¹⁵, broad-line S/N
10.5, FWHM quartiles 5900–8200 km/s with **nothing railed**.  Hα carries 540
of the 942 detections, then Hβ 189, He I 76, Hγ 50, Hδ 46, Pa-β 31.

This is the AGN completeness curve the project needed, measured on real AGN
rather than injections, on a sample 21× larger than yesterday's.

### Detection rate is set by which line the grism can reach

Splitting the 2,379 confirmed quasars by whether broad Hα falls inside
12500–18500 Å explains most of the spread:

| | n | detected |
|---|---:|---:|
| Hα **in** the grism | 1,023 | **54 %** |
| Hα outside it | 1,357 | 29 % |
| Hα in range **and** S/N > 10 | 87 | **80 %** |

By the line the fit used: Hα 56 %, He I 10830 48 %, Pa-β 46 %, Hβ 29 %,
Pa-γ 24 %, Hδ 21 %, Hγ 19 % — and by redshift, Hα availability is 100 % at
z 0.9–1.5 (56 % detected), 77 % at 1.5–1.9 (43 %), zero below 0.9 (44 %) and
zero above 1.9 (28 % falling to 22 %).

So the non-detections are mostly physics, not pipeline failure: outside
0.9 < z < 1.9 the grism sees only Pa-γ, Pa-β, He I or the higher Balmer
lines, which are 10–50× weaker than Hα in a quasar.  The honest headline is
therefore **54 % of confirmed quasars are detected where Hα is observable,
rising to 80 % at S/N > 10**, and the AGN survey's natural redshift window is
0.9 < z < 1.9.

### Gallery of the decomposition (`plots/agn_gallery/`)

Two subfolders, because the failures teach as much as the successes:
``detected/`` sorted by Δχ² (strongest first) and ``not_detected/`` sorted by
S/N descending, so the first pages are bright confirmed quasars where no
broad line was recovered.  Each panel draws the spectrum with its ±1σ band,
the full model, the continuum alone and continuum+broad, so it is visible
whether the fit is describing a line or reshaping the continuum.

Reading page 1 of each: the strong detections are textbook broad lines with
the continuum correctly beneath, but several high-Δχ² "detections" are
artefact-driven (chaotic spectra, single-pixel spikes) — the same lesson as
the NGC 1527 velocity pile-up, and an argument for running the artefact cut
before the AGN statistics rather than after.

### Both failure modes in the gallery, diagnosed

Visual inspection showed spurious detections and obvious missed lines.  Both
are now measured (`validation/agn_diagnostics.py`).

**Spurious detections are spikes.**  A σ = 3000 km/s line covers ~25 pixels; a
cosmic ray or decontamination edge covers one to three.  The fraction of Δχ²
carried by the three best pixels separates them cleanly: 90 % of the signal
sits in a median of **19 pixels for real detections and 3 for spike-like
ones**.  Of the 942 detections, 132 (14 %) have > 50 % of Δχ² in three pixels,
43 (5 %) over 70 %, 23 (2 %) over 90 %.  Examples:
`plots/agn_spurious_detections.png`.

**Missed lines are my own orthogonality guard.**  Searching the residual for
unmodelled emission first found mostly *broad artefacts* — contamination
blocks and edge steps, whose rest wavelengths cluster at no transition
(`plots/agn_missed_lines.png`).  Searching instead at the *expected line
position* found the real thing: **35 % of non-detections with S/N > 3 carry a
> 5 σ flux excess over the continuum at the line**, 7 % over 10 σ.  For the
nine strongest, containment passes 5–8 of 8 widths but **orthogonality ≥ 0.8
passes only 1–5**, so the wide components a quasar actually has are never
tried:

| object | line | Δχ² in production | Δχ² with widths unguarded |
|---|---|---:|---:|
| 2707716761636953296 | Hβ | 22 | **1233** |
| 2683286183647528784 | Hβ | 11 | **276** |
| 2720080134671460552 | Hα | 10 | **165** |
| 2734824261663319750 | Hα | 6 | **107** |

The two errors have opposite signs and one root cause: **a hard orthogonality
cut is the wrong instrument.**  Yesterday it was calibrated to stop width
railing, and it does — by discarding the widest components whether they are
real or not.  Wide components are genuinely more degenerate with the
continuum, so they should carry a *higher Δχ² threshold*, not be excluded.

Planned fix, in order: (1) width-dependent Δχ² thresholds calibrated from the
off-redshift null, replacing the hard orthogonality cut; (2) a Δχ²-
concentration cut to remove spike-driven detections; (3) re-run and re-quote
completeness and purity.  Until then the 40 %/54 % detection figures are a
*lower* bound and the purity figures an upper one.
