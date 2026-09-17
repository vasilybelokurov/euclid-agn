# plots/

PNG figures, all regenerable:

```bash
export PYTHONPATH=src
python -m euclid_agn plot \
    --screen outputs/screen_pilot_v3.parquet \
    --glob '~/data/euclid/q1/SIR/*/*.fits' \
    --directory plots --n-candidates 8
```

| file | what it shows |
|---|---|
| `candidate_grid.png` | the top Stage-1 candidates, zoomed on the broad line under test. Grey data, blue M0 (continuum + narrow), red M1 (all components), dashed red M1 continuum + broad only. |
| `screen_statistics.png` | where the detection statistic sits in the population, the M0 goodness of fit against the independently measured noise inflation, the statistic against continuum orthogonality, and the recovered widths. |
| `availability_sample.png` | the Q1 parent sample: usable pixel fraction, continuum S/N, effective LSF and dither count over 1214 real spectra. |
| `completeness_desi_halpha.png` | broad-Hα injection/recovery into 38 real DESI hosts: the empirical null, completeness against injected flux and width at 5 % and 1 % FPR, and the flux bias of detections. |
| `confident_wrong_desi.png`, `confident_wrong_dithers.png` | the confident-but-wrong DESI identifications: strong features at no catalogue line, shown per dither to be present in one orientation only — slitless contamination. |
| `desi_redshift_recovery.png` | pipeline redshift against DESI for 114 Hα-window galaxies, with agreement against the detectability of Hα at the DESI redshift. |
| `high_statistic_objects.png` | the six spectra with the largest identification statistics before outlier rejection — all driven by unmasked pixel spikes. |
| `confident_wrong.png` | seven confident identifications that disagreed with SPE; three are SPE errors, one is a broad-line candidate. |
| `fits/fit_<object_id>.png` | one candidate in full: data, both models, the broad component on the M1 continuum, and the normalised residuals of both models. |
| `spectra/spectrum_<object_id>.png` | one object's combined spectrum above each contributing dither, with unusable pixels shaded and contaminant counts labelled. |

## What the grey shading means

| figure | grey element | meaning |
|---|---|---|
| `spectra/*.png` | mid-grey, full height, one pixel wide | **unusable pixel**: SIR `MASK` has the `NOT_USE` bit, or the variance is non-positive or non-finite, or the flux is non-finite. Dropped from every fit, shown rather than hidden. |
| `spectra/*.png` | very light grey at both ends | outside the 12500–18500 Å science window. The stored grid runs 11900–19002 Å. |
| `fits/*.png`, `candidate_grid.png` | band hugging the data | **±1σ from the reported per-pixel variance** (`VAR × FSCALE²`). |
| `screen_statistics.png` | grey histogram | all rows; the blue overlay is the subset with reduced χ²(M0) < 4. |

The ±1σ band is the *reported* variance, which the noise audit measured to be
about 1.45× too small, so the true scatter is wider than the band looks. That is
why residual panels routinely run past ±2σ.

Two reference objects are included for orientation:

- `spectra/spectrum_2731173428682078045.png` — the IRSA tutorial object, an
  extended galaxy that is 98.3 % masked, with nine usable pixels at the red edge;
- `spectra/spectrum_2734482961680140786.png` — a compact, high signal-to-noise
  source, where the per-dither panels show narrow spikes that differ between
  dithers and largely vanish on co-addition.

**Nothing in `fits/` is a detection.** The Stage-1 statistic is uncalibrated;
`screen_statistics.png` shows that a nominal Δχ² = 25 sits near the 95th
percentile of unselected spectra. Reading these figures is how three separate
failure modes were found — see `JOURNAL.md`.
