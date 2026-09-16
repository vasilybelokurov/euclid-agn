"""Shared fixtures.  Unit tests never touch the network."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

REFERENCE_OBJECT_ID = 2731173428682078045
REFERENCE_TILE_ID = 102160339


@pytest.fixture(scope="session")
def sir_test_file(tmp_path_factory) -> Path:
    """A small SIR-format FITS file written by the simulator."""
    from euclid_agn.validation.simulator import make_test_file

    path = tmp_path_factory.mktemp("sir") / "EUC_SIR_W-COMBSPEC_999999999_sim.fits"
    return make_test_file(path, n_objects=3)
