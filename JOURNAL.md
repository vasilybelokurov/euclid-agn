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
