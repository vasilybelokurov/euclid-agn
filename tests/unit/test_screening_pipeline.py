"""Stage-1 production loop, driven by a simulated archive file."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from euclid_agn.config import Config
from euclid_agn.pipeline.screening import (
    CONTEXT_COLUMNS,
    hypotheses_for,
    screen_file,
    screen_manifest,
    settings_from_config,
)
from euclid_agn.validation.simulator import (
    LineTruth,
    SpectrumTruth,
    simulate_observation,
    write_sir_file,
)


@pytest.fixture(scope="module")
def archive(tmp_path_factory):
    """A three-object SIR file: narrow-only, broad, and mostly masked."""
    directory = tmp_path_factory.mktemp("screen")
    narrow = (LineTruth("Halpha", 4e-16, 150.0), LineTruth("NII6584", 1.2e-16, 150.0))
    observations = [
        simulate_observation(SpectrumTruth(z=1.2, lines=narrow, object_id=101, seed=1)),
        simulate_observation(
            SpectrumTruth(
                z=1.2,
                lines=(*narrow, LineTruth("Halpha", 2.5e-15, 2500.0, broad=True)),
                object_id=102,
                seed=2,
            )
        ),
        simulate_observation(
            SpectrumTruth(z=1.2, lines=narrow, object_id=103, seed=3, masked_fraction=0.9)
        ),
    ]
    path = write_sir_file(directory / "EUC_SIR_W-COMBSPEC_1_sim.fits", observations)
    manifest = pd.DataFrame(
        {
            "object_id": [101, 102, 103],
            "ra": [270.0, 270.1, 270.2],
            "dec": [66.0, 66.1, 66.2],
            "tile_id": [999999999] * 3,
            "release": ["sim"] * 3,
            "sir_s3_key": [str(path)] * 3,
            "sir_hdu": [2, 2, 2],
            "spe_gal_z": [1.2, 1.2, 1.2],
            "spe_class": ["galaxy"] * 3,
            "flux_h_2fwhm_aper": [50.0, 60.0, 70.0],
        }
    )
    manifest_path = directory / "manifest.parquet"
    manifest.to_parquet(manifest_path, index=False)
    return path, manifest, manifest_path


def fast_config() -> Config:
    return Config(
        screening={"redshift_sources": ["spe"], "blind_z_step_kms": 3000.0},
        broad={"sigma_grid_n": 3, "sigma_min_kms": 500.0, "sigma_max_kms": 4000.0},
        selection={"min_usable_pixel_fraction": 0.5},
    )


def test_settings_follow_the_config():
    config = Config(continuum={"n_knots": 5}, broad={"sigma_grid_n": 4})
    settings = settings_from_config(config)
    assert settings.n_knots == 5
    assert len(settings.broad_sigma_kms) == 4


def test_hypotheses_fall_back_to_the_catalogue_without_a_spectrum():
    row = pd.Series({"spe_gal_z": 1.31})
    hypotheses = hypotheses_for(row, fast_config(), spectrum=None)
    assert [h.origin for h in hypotheses] == ["spe_galaxy"]


def test_screen_file_skips_the_mostly_masked_object(archive):
    path, manifest, _ = archive
    table = screen_file(str(path), manifest, fast_config())
    assert set(table["object_id"]) <= {101, 102}
    assert 103 not in set(table["object_id"])


def test_screen_file_carries_context_and_noise_covariates(archive):
    path, manifest, _ = archive
    table = screen_file(str(path), manifest, fast_config())
    row = table.iloc[0]
    for column in ("ra", "dec", "tile_id", "spe_class", "flux_h_2fwhm_aper"):
        assert column in table.columns, column
    assert set(CONTEXT_COLUMNS) & set(table.columns)
    assert row["noise_inflation"] >= 1.0
    assert "noise_acf_lag1" in table.columns


def test_broad_object_scores_above_the_narrow_only_object(archive):
    path, manifest, _ = archive
    table = screen_file(str(path), manifest, fast_config())
    scores = table.set_index("object_id")["delta_chi2"]
    assert scores.loc[102] > scores.loc[101]
    assert table.set_index("object_id").loc[102, "broad_flux_Halpha"] > 0


def test_screen_manifest_writes_results_shards_and_provenance(archive, tmp_path):
    import json

    _, _, manifest_path = archive
    output = tmp_path / "screen_results.parquet"

    class LocalBackend:
        class _Cache:
            @staticmethod
            def fetch(key, filesystem, force=False):
                return Path(key)

        cache = _Cache()
        filesystem = None

    results = screen_manifest(fast_config(), manifest_path, output, backend=LocalBackend())
    assert output.exists()
    assert not results.empty
    shards = list((tmp_path / "screen_results_shards").glob("*.parquet"))
    assert len(shards) == 1
    payload = json.loads((tmp_path / "run_manifest.json").read_text())
    assert payload["extra"]["n_rows"] == len(results)
    assert payload["config_checksum"]
    assert payload["line_list_version"]


def test_screen_manifest_reuses_shards_on_rerun(archive, tmp_path):
    _, _, manifest_path = archive
    output = tmp_path / "again.parquet"

    class CountingBackend:
        def __init__(self):
            self.calls = 0

        class _Cache:
            def __init__(self, outer):
                self.outer = outer

            def fetch(self, key, filesystem, force=False):
                self.outer.calls += 1
                return Path(key)

        filesystem = None

        @property
        def cache(self):
            return CountingBackend._Cache(self)

    backend = CountingBackend()
    screen_manifest(fast_config(), manifest_path, output, backend=backend)
    first = backend.calls
    screen_manifest(fast_config(), manifest_path, output, backend=backend)
    assert backend.calls == first


def test_screen_manifest_rejects_an_empty_selection(archive, tmp_path):
    _, _, manifest_path = archive
    with pytest.raises(ValueError):
        screen_manifest(
            fast_config(), manifest_path, tmp_path / "x.parquet", object_ids=[999999]
        )


def test_screen_file_puts_the_noise_scale_into_the_likelihood(archive):
    """With noise_scale_free the variance is rescaled before fitting."""
    path, manifest, _ = archive
    on = screen_file(str(path), manifest, fast_config())
    off_config = fast_config()
    off_config.screening.noise_scale_free = False
    off = screen_file(str(path), manifest, off_config)
    assert "noise_variance_scale" in on.columns
    assert (off["noise_variance_scale"] == 1.0).all()
    # Rescaling the variance by s^2 divides every chi-squared by s^2.
    joined = on.merge(off, on="object_id", suffixes=("_on", "_off"))
    scale = joined["noise_variance_scale_on"]
    assert np.allclose(joined["chi2_m0_on"] * scale, joined["chi2_m0_off"], rtol=1e-6)
