import numpy as np
import pytest

from euclid_agn.constants import SIR_BINWIDTH_ANGSTROM
from euclid_agn.models.broad import (
    BroadComponent,
    BroadFamily,
    forbidden_broad_basis,
    sigma_grid,
)
from euclid_agn.validation.simulator import sir_wavelength_grid

Z = 1.2
LSF = 13.7


def component(**kwargs) -> BroadComponent:
    base = dict(line_name="Halpha", z=Z, sigma_kms=2000.0, lsf_sigma=LSF)
    base.update(kwargs)
    return BroadComponent(**base)


def test_forbidden_transitions_are_refused_by_default():
    with pytest.raises(ValueError, match="forbidden"):
        BroadComponent(line_name="OIII5007", z=Z, sigma_kms=2000.0, lsf_sigma=LSF)


def test_forbidden_transitions_are_available_for_null_tests_only():
    c = BroadComponent(
        line_name="OIII5007", z=Z, sigma_kms=2000.0, lsf_sigma=LSF, allow_forbidden=True
    )
    assert not c.line.permitted
    basis = forbidden_broad_basis("OIII5007", sir_wavelength_grid(), 1.8, 2000.0, LSF)
    assert np.all(basis >= 0)


def test_permitted_transitions_are_allowed():
    for name in ("Halpha", "Hbeta", "Pabeta", "MgII2796", "HeI10830"):
        assert BroadComponent(line_name=name, z=0.3, sigma_kms=1500.0, lsf_sigma=LSF)


def test_bad_widths_are_rejected():
    with pytest.raises(ValueError):
        component(sigma_kms=0.0)
    with pytest.raises(ValueError):
        component(lsf_sigma=-1.0)
    with pytest.raises(ValueError):
        component(line_name="NotALine")


def test_widths_add_in_quadrature_with_the_lsf():
    c = component(sigma_kms=2000.0)
    assert c.observed_sigma_angstrom == pytest.approx(
        np.hypot(c.intrinsic_sigma_angstrom, LSF)
    )
    assert c.observed_sigma_angstrom > c.intrinsic_sigma_angstrom


def test_fwhm_and_resolution_ratio():
    c = component(sigma_kms=2000.0)
    assert c.fwhm_kms == pytest.approx(2.3548 * 2000.0, rel=1e-3)
    # sigma 2000 km/s at 14444 A is ~96 A, about 7 LSF sigmas.
    assert c.resolution_ratio == pytest.approx(7.0, rel=0.1)
    assert component(sigma_kms=200.0).resolution_ratio < 1.0


def test_basis_is_unit_flux_and_broader_than_a_narrow_line():
    w = sir_wavelength_grid()
    broad = component(sigma_kms=3000.0).basis(w, SIR_BINWIDTH_ANGSTROM)
    narrow = component(sigma_kms=100.0).basis(w, SIR_BINWIDTH_ANGSTROM)
    assert broad.sum() * SIR_BINWIDTH_ANGSTROM == pytest.approx(1.0, rel=1e-4)
    assert narrow.sum() * SIR_BINWIDTH_ANGSTROM == pytest.approx(1.0, rel=1e-4)
    assert broad.max() < narrow.max()


def test_velocity_offset_moves_the_centre():
    blue = component(velocity_kms=-1000.0).centre
    red = component(velocity_kms=+1000.0).centre
    assert blue < component().centre < red


def test_in_range_detects_a_line_outside_the_grid():
    w = sir_wavelength_grid()
    assert component().in_range(w)
    # Halpha at z=3 is at 26258 A, far beyond the red grism.
    assert not component(z=3.0).in_range(w)


def test_family_shares_width_but_not_flux():
    family = BroadFamily(("Halpha", "Hbeta"), z=1.7, sigma_kms=2500.0, lsf_sigma=LSF)
    components = family.components()
    assert {c.sigma_kms for c in components} == {2500.0}
    assert components[0].centre != components[1].centre
    design = family.design(sir_wavelength_grid(), SIR_BINWIDTH_ANGSTROM)
    assert design.shape == (531, 2)
    assert family.names == ("broad_Halpha", "broad_Hbeta")


def test_family_reports_which_components_are_observable():
    family = BroadFamily(("Halpha", "Hbeta"), z=1.2, sigma_kms=2000.0, lsf_sigma=LSF)
    visible = family.visible_components(sir_wavelength_grid())
    assert [c.line_name for c in visible] == ["Halpha"]


def test_sigma_grid_is_logarithmic_and_bounded():
    grid = sigma_grid(300.0, 6000.0, 12)
    assert grid.size == 12
    assert grid[0] == pytest.approx(300.0)
    assert grid[-1] == pytest.approx(6000.0)
    ratios = grid[1:] / grid[:-1]
    assert np.allclose(ratios, ratios[0])
    with pytest.raises(ValueError):
        sigma_grid(600.0, 300.0)
