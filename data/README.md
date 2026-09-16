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
