# euclid-agn — host-unbiased AGN search in Euclid NISP spectra

Discover AGN in public Euclid NISP slitless spectra by emission-line
decomposition (Chilingarian et al. 2018, [arXiv:1805.01467](https://arxiv.org/abs/1805.01467)),
adapted to Euclid's morphology-dependent line-spread function, multiple dithers,
contamination and limited wavelength coverage.

Candidates are selected from the spectra alone: no cut on host stellar mass,
luminosity, colour, morphology, star-formation state or existing AGN
classification. Host and data properties are recorded so the selection function
can be reported against them.

## Status

| Milestone | State |
|---|---|
| M0 package foundation | done |
| M1 Q1 archive layer | done (IRSA backend; ESA backend declared, not implemented) |
| M2 simulator and model primitives | partial — line catalogue, LSF, simulator done; continuum/NLR/BLR solvers pending |
| M3 Stage-1 fitter | not started |
| M4 EDF-N validation | not started |
| M5 joint dither fitting | not started |
| M6 Q1 production run | not started |

No AGN detection claim is possible yet: nothing has been fitted, so nothing has
been validated.

## Install and run

```bash
source ~/Work/venvs/.venv/bin/activate
cd ~/Work/Code/euclid_spectral_fit
export PYTHONPATH=src            # or: pip install -e .

python -m euclid_agn version
python -m euclid_agn archive list-tiles EDF-N
python -m euclid_agn spectra inspect --object-id 2731173428682078045 --plot outputs/ref.png
python -m euclid_agn archive build-manifest \
    --config configs/q1.yaml --tile 102160339 --limit 200 \
    --output data/manifests/q1_tile102160339.parquet
```

## Tests

```bash
export PYTHONPATH=src
python -m pytest tests -q              # unit + regression, no network
python -m pytest tests/online -m online -q   # opt-in, hits IRSA
```

`tests/regression` runs against a 222 kB frozen extract of two real Q1 objects,
so the archive representation is pinned without any network access.

## What was measured, not assumed

Everything below was checked against live Q1 products on 2026-09-16 and is
recorded in `JOURNAL.md` with the command used.

- SIR file layout, per-object HDU grouping and the varying HDU count per object.
- `FSCALE` convention: `flux = SIGNAL * FSCALE`, `variance = VAR * FSCALE**2`.
- Mask bits behave as a bit field (values 1, 64, 65, 66, 67 seen in real data).
- `LSF_SIG` distribution: median 13.7 Å, i.e. R ≈ 465 at 1.5 µm for compact
  sources, with a tail to R ≈ 90 for extended ones.
- Wavelength grid: 531 bins of 13.4 Å from 11900 Å — about one LSF sigma per
  bin, so models are integrated over pixels rather than sampled at centres.
- Rest wavelengths are **vacuum** (median implied H-alpha rest wavelength
  6564.39 Å from 400 Q1 SPE detections; vacuum 6564.61, air 6562.80).
- The four Q1 field cones select 352 tiles, exactly the number of distinct Q1
  tiles, so they partition the release.
- Masking dominates usability: the IRSA tutorial reference object
  (2731173428682078045) is 98.3 % `NOT_USE`-masked, leaving nine pixels at the
  red edge, while a compact neighbour keeps 98.9 % of its spectrum. Selecting on
  signal-to-noise alone would accept the first one.

## Layout

```
src/euclid_agn/
  archive/     TAP client, IRSA backend, ESA placeholder, table schema
  io/          SIR FITS reader (the only place archive units are converted)
  spectra/     immutable spectrum types, mask handling, LSF
  models/      line catalogue and (pending) spectral components
  fit/         (pending) linear solvers, hypotheses, screening, statistics
  validation/  simulator and SIR-format writer; (pending) injections, metrics
  pipeline/    manifest ingest; (pending) screening, refinement, catalogue
  plotting/    diagnostics
```

Core science never lives only in a notebook. Raw FITS are immutable; derived
data are reproducible; every run writes `run_manifest.json`.
