"""Atlas plotting: figures must be produced, and must not invent data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from euclid_agn.fit.screen import ScreenSettings, screen_spectrum
from euclid_agn.fit.hypotheses import RedshiftHypothesis
from euclid_agn.plotting.atlas import (
    build_atlas,
    plot_candidate,
    plot_fit_grid,
    plot_screen_summary,
    plot_spectrum_page,
)
from euclid_agn.validation.simulator import (
    LineTruth,
    SpectrumTruth,
    simulate_observation,
    write_sir_file,
)

SETTINGS = ScreenSettings(broad_sigma_kms=(500.0, 1200.0), n_refine=1)


@pytest.fixture(scope="module")
def archive(tmp_path_factory):
    directory = tmp_path_factory.mktemp("atlas")
    narrow = (LineTruth("Halpha", 4e-16, 150.0), LineTruth("NII6584", 1.2e-16, 150.0))
    observations = [
        simulate_observation(
            SpectrumTruth(
                z=1.2,
                lines=(*narrow, LineTruth("Halpha", 2.5e-15, 1200.0, broad=True)),
                object_id=201,
                seed=1,
            )
        ),
        simulate_observation(SpectrumTruth(z=1.2, lines=narrow, object_id=202, seed=2)),
    ]
    path = write_sir_file(directory / "EUC_SIR_W-COMBSPEC_2_sim.fits", observations)
    return [str(path)]


@pytest.fixture(scope="module")
def screen_table(archive):
    from euclid_agn.io.sir import open_sir_file

    rows = []
    with open_sir_file(archive[0]) as sir:
        for group in sir.groups().values():
            spectrum = sir.read_combined(group)
            table = screen_spectrum(
                spectrum,
                [RedshiftHypothesis(1.2, "spe_galaxy", "halpha_complex")],
                SETTINGS,
                object_id=group.object_id,
                noise_inflation=1.4,
            )
            if not table.empty:
                rows.append(table.head(1))
    return pd.concat(rows, ignore_index=True)


def test_candidate_figure_is_written(screen_table, archive, tmp_path):
    row = screen_table.iloc[0]
    path = plot_candidate(row, archive, tmp_path, SETTINGS)
    assert path is not None and path.exists()
    assert path.suffix == ".png"
    assert path.stat().st_size > 5000


def test_candidate_figure_returns_none_for_an_unknown_object(screen_table, archive, tmp_path):
    row = screen_table.iloc[0].copy()
    row["object_id"] = 999999
    assert plot_candidate(row, archive, tmp_path, SETTINGS) is None


def test_spectrum_page_is_written(archive, tmp_path):
    path = plot_spectrum_page(201, archive, tmp_path)
    assert path is not None and path.exists()
    assert plot_spectrum_page(999999, archive, tmp_path) is None


def test_grid_handles_more_panels_than_candidates(screen_table, archive, tmp_path):
    path = plot_fit_grid(screen_table.head(1), archive, tmp_path / "grid.png", SETTINGS)
    assert path.exists() and path.stat().st_size > 5000


def test_summary_figure_is_written(screen_table, tmp_path):
    path = plot_screen_summary(screen_table, tmp_path / "summary.png")
    assert path.exists() and path.stat().st_size > 5000


def test_build_atlas_creates_the_expected_tree(screen_table, archive, tmp_path):
    created = build_atlas(screen_table, archive, directory=tmp_path, n_candidates=2)
    assert (tmp_path / "fits").is_dir()
    assert (tmp_path / "spectra").is_dir()
    assert (tmp_path / "candidate_grid.png").exists()
    assert (tmp_path / "screen_statistics.png").exists()
    assert len(created["fits"]) >= 1
    assert all(Path(p).suffix == ".png" for group in created.values() for p in group)


def test_atlas_ranks_objects_with_a_usable_continuum_model_first(screen_table, archive, tmp_path):
    table = screen_table.copy()
    table.loc[:, "delta_chi2_refined_effective"] = [10.0, 500.0][: len(table)]
    table.loc[:, "continuum_model_ok"] = [True, False][: len(table)]
    created = build_atlas(table, archive, directory=tmp_path, n_candidates=1, with_dithers=False)
    # The acceptable-continuum object wins despite the smaller statistic.
    assert str(int(table.iloc[0]["object_id"])) in created["fits"][0]


def test_broad_overlay_uses_the_m1_continuum_not_the_m0_one(screen_table, archive):
    """The dashed overlay must be M1 minus its narrow lines, nothing else."""
    import numpy as np

    from euclid_agn.io.sir import open_sir_file
    from euclid_agn.fit.screen import refine
    from euclid_agn.plotting.diagnostics import broad_on_continuum

    row = screen_table.iloc[0]
    with open_sir_file(archive[0]) as sir:
        spectrum = sir.read_combined(sir.group_for_object(int(row["object_id"])))
    fit = refine(spectrum, float(row["z"]), str(row["system"]), 1200.0, settings=SETTINGS)
    overlay = broad_on_continuum(fit)
    narrow_block = fit.blocks_m1.design[:, fit.blocks_m1.slices["narrow"]]
    narrow_model = narrow_block @ fit.blocks_m1.block(fit.m1, "narrow")
    # The invariant: the overlay plus the narrow lines reconstructs M1 exactly.
    assert np.allclose(overlay + narrow_model, fit.m1.model, rtol=1e-8, atol=1e-30)

    # The naive alternative - M0's model plus M1's broad block - is not part of
    # any consistent decomposition, because M1 refits the continuum. On this
    # clean synthetic spectrum the two continua happen to agree closely, so the
    # test asserts the structural fact rather than a numerical difference.
    broad_block = fit.blocks_m1.design[:, fit.blocks_m1.slices["broad"]]
    naive = fit.m0.model + broad_block @ fit.blocks_m1.block(fit.m1, "broad")
    assert not np.allclose(naive + narrow_model, fit.m1.model, rtol=1e-8, atol=1e-30)


def test_spectrum_page_labels_its_shading(archive, tmp_path):
    """Grey shading must be explained on the figure, not just in the docs."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from euclid_agn.io.sir import open_sir_file
    from euclid_agn.plotting.diagnostics import plot_spectrum

    with open_sir_file(archive[0]) as sir:
        spectrum = sir.read_combined(sir.group_for_object(201))
    fig, ax = plt.subplots()
    plot_spectrum(ax, spectrum)
    labels = ax.get_legend_handles_labels()[1]
    plt.close(fig)
    joined = " ".join(labels).lower()
    assert "unusable" in joined
    assert "outside" in joined
    assert "sigma" in joined or "\\sigma" in joined
