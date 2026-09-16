# Project specification

The authoritative brief lives in the repository root `CLAUDE.md` /
`AGENTS.md` conversation record. This file records only deviations and
decisions taken while implementing it.

## Deviations from the original brief

| Brief | Implementation | Reason |
|---|---|---|
| package name `euclid-agn`, directory `euclid-agn/` | package `euclid_agn` inside the existing `euclid_spectral_fit/` working directory | the working directory already existed and is the user's project root |
| `uv.lock`, `uv` workflow | plain `pyproject.toml` + the shared venv at `~/Work/venvs/.venv` | `uv` is not installed on this machine; `ruff`, `hypothesis` and `polars` are also absent, so they are optional extras |
| field selection by MER cone/box query | field → tile list through the CAOM tables | a `DISTINCT tileid` box query on MER fails after 5 minutes on the live service; the CAOM route returns in ~15 s |
| server-side joins of MER with the association table | separate queries joined client-side | IRSA serves them from different datasources and rejects the join |
| air/unspecified rest wavelengths in the brief's coverage table | vacuum rest wavelengths | measured against 400 Q1 SPE H-α detections |

## Open questions

- Variance correlation between neighbouring pixels after SIR resampling. The
  adjacent-difference scatter is ~1.25× √VAR (median over 973 spectra), which
  bounds but does not identify the pixel covariance. Detection significance
  must be calibrated empirically rather than read off a χ² distribution.
- `QUALITY` semantics: values are floats in [0, 1] and their exact definition
  is not yet pinned to a Euclid document.
- Whether `LSF_SIG` alone captures the morphology dependence well enough for
  broad-line width recovery, or whether Stage 3 forward modelling is needed.
