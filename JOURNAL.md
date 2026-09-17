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
