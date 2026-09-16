import pytest

from euclid_agn.models.line_catalog import (
    BY_NAME,
    FIXED_RATIOS,
    LINES,
    coverage,
    redshift_range,
    visible,
    visible_lines,
    visible_systems,
)


def test_names_are_unique():
    assert len({line.name for line in LINES}) == len(LINES)


def test_wavelengths_are_vacuum():
    # Measured from 400 Q1 SPE Halpha detections: median implied rest
    # wavelength 6564.39 A (vacuum 6564.61, air 6562.80).
    assert BY_NAME["Halpha"].rest == pytest.approx(6564.61, abs=0.05)
    assert BY_NAME["Hbeta"].rest == pytest.approx(4862.68, abs=0.05)


def test_forbidden_lines_are_not_permitted():
    for name in ("NII6584", "OIII5007", "SII6716", "OII3726", "SIII9531"):
        assert not BY_NAME[name].permitted
    for name in ("Halpha", "Hbeta", "Pabeta", "MgII2796", "HeI10830"):
        assert BY_NAME[name].permitted


def test_redshift_ranges_match_the_q1_red_grism():
    lo, hi = redshift_range("Halpha")
    assert lo == pytest.approx(0.904, abs=0.01)
    assert hi == pytest.approx(1.818, abs=0.01)
    lo, hi = redshift_range("Pabeta")
    assert lo == pytest.approx(-0.025, abs=0.01)
    assert hi == pytest.approx(0.443, abs=0.01)


def test_visibility_is_consistent_with_the_range():
    lo, hi = redshift_range("Halpha")
    assert visible("Halpha", 0.5 * (lo + hi))
    assert not visible("Halpha", hi + 0.05)
    assert not visible("Halpha", lo - 0.05)


def test_bpt_set_is_complete_only_in_a_narrow_window():
    assert coverage(1.7).bpt_complete
    assert not coverage(1.2).bpt_complete
    assert not coverage(2.2).bpt_complete


def test_broad_branch_is_available_far_outside_the_bpt_window():
    # The whole point of the BLR branch: it works where BPT cannot.
    for z in (0.2, 0.5, 1.2, 2.5, 4.0):
        assert coverage(z).broad_testable, f"no permitted line visible at z={z}"


def test_visible_systems_require_broad_member_when_asked():
    systems = visible_systems(1.2, require_broad=True)
    assert "halpha_complex" in {s.name for s in systems}


def test_visible_lines_at_low_redshift_are_paschen():
    names = {line.name for line in visible_lines(0.1)}
    assert "Pabeta" in names
    assert "Halpha" not in names


def test_fixed_ratios_reference_known_lines_and_are_physical():
    for (strong, weak), ratio in FIXED_RATIOS.items():
        assert strong in BY_NAME and weak in BY_NAME
        assert 0.0 < ratio < 1.0
        assert BY_NAME[weak].rest < BY_NAME[strong].rest
