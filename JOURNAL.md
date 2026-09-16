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
