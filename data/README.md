# data/

Derived, reproducible products only. Raw Euclid FITS live in the archive and are
never copied here in bulk.

```
manifests/<name>.parquet         one row per source with a Q1 spectrum
manifests/<name>_shards/         per-tile shards; a rerun reuses them
manifests/run_manifest.json      provenance for the last manifest build
```

Rebuild any manifest with:

```bash
export PYTHONPATH=src
python -m euclid_agn archive build-manifest \
    --config configs/q1.yaml --tile <TILE_ID> \
    --output data/manifests/<name>.parquet
```

The manifest carries MER photometry and morphology, PHZ redshifts and SPE
class/redshift/quality. All of it is **context**: it is reported and used to
stratify the selection function, never to select AGN candidates.

## Where things live (reorganised 2026-09-19)

Spectral template libraries are **generic** — the same files serve any project that fits
spectra — so they sit outside the Euclid tree:

    ~/data/spectral_templates/
        phoenix/     PHOENIX ACES R=10000 synthetic stars (5.6 GB, Husser et al. 2013)
        xsl_dr3/     X-shooter Spectral Library DR3, 830 observed stars (1.4 GB)
        xsl_ssp/     XSL simple stellar populations (364 MB, Verro et al. 2022)
        qso/         Glikman et al. 2006 quasar composite (352 kB)
        README.md    provenance and download commands

Override the location with the ``SPECTRAL_TEMPLATES`` environment variable;
``euclid_agn.models.library.DEFAULT_ROOT`` reads it.

Euclid-specific data and derived products stay under ``~/data/euclid/``:

    q1/SIR/          the Q1 slitless spectra themselves
    with_desi/       the DESI cross-match file
    rvspecfit/       PHOENIX convolved to the NISP LSF (R = lambda/32.3, 11500-19500 A)
                     plus interpolator and CCF files - useless outside NISP and
                     regenerable in ~10 min by scripts/rvspecfit_nisp_setup.sh
