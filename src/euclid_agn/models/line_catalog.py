"""Rest-frame line catalogue and redshift visibility.

Wavelength convention
---------------------
All rest wavelengths are **vacuum**, in Angstrom.

This is not an assumption: it was measured.  Taking 2000 Q1 SPE H-alpha
detections with ``spe_line_snr_gf > 10`` and rank-0 galaxy redshifts, the
implied rest wavelength ``lambda_obs / (1 + z)`` has median 6564.39 A
(400 matched objects), against 6564.61 A vacuum and 6562.80 A air.  The
interquartile range is 6559.5-6569.8 A, so an individual line does not decide
the question, but the median is 0.2 A from vacuum and 1.6 A from air.

Line-list version is tracked in :data:`euclid_agn.provenance.LINE_LIST_VERSION`;
bump it whenever this file changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from euclid_agn.constants import RGS_SCIENCE_WMAX_ANGSTROM, RGS_SCIENCE_WMIN_ANGSTROM


@dataclass(frozen=True)
class Line:
    """One transition.

    Parameters
    ----------
    name : str
        Unique key, e.g. ``"Halpha"``.
    rest : float
        Vacuum rest wavelength, Angstrom.
    permitted : bool
        Whether a broad (BLR) component is physically allowed.  Forbidden
        transitions must never carry a broad component in the AGN model; the
        fitter may be *asked* to fit one only as a null test.
    family : str
        Group of lines expected to share a redshift and narrow profile.
    label : str
        Display label.
    """

    name: str
    rest: float
    permitted: bool
    family: str
    label: str = ""

    def observed(self, z: float, velocity_kms: float = 0.0) -> float:
        """Observed wavelength at redshift ``z`` with an extra velocity offset."""
        from euclid_agn.constants import C_KMS

        return self.rest * (1.0 + z) * (1.0 + velocity_kms / C_KMS)


#: Vacuum rest wavelengths, Angstrom.
LINES: tuple[Line, ...] = (
    # Hydrogen Balmer (permitted)
    Line("Halpha", 6564.61, True, "balmer", r"H$\alpha$"),
    Line("Hbeta", 4862.68, True, "balmer", r"H$\beta$"),
    Line("Hgamma", 4341.68, True, "balmer", r"H$\gamma$"),
    Line("Hdelta", 4102.89, True, "balmer", r"H$\delta$"),
    # Hydrogen Paschen (permitted) - the low-z AGN probe inside the red grism
    Line("Paalpha", 18756.13, True, "paschen", r"Pa$\alpha$"),
    Line("Pabeta", 12821.58, True, "paschen", r"Pa$\beta$"),
    Line("Pagamma", 10941.09, True, "paschen", r"Pa$\gamma$"),
    Line("Padelta", 10052.13, True, "paschen", r"Pa$\delta$"),
    # Helium (permitted)
    Line("HeI10830", 10833.31, True, "helium", "He I 10830"),
    Line("HeII4686", 4687.02, True, "helium", "He II 4686"),
    # Ultraviolet permitted lines, reachable at high z
    Line("MgII2796", 2796.35, True, "mgii", "Mg II 2796"),
    Line("MgII2803", 2803.53, True, "mgii", "Mg II 2803"),
    Line("CIII1909", 1908.73, True, "ciii", "C III] 1909"),
    Line("CIV1548", 1548.19, True, "civ", "C IV 1548"),
    Line("CIV1551", 1550.77, True, "civ", "C IV 1551"),
    Line("Lyalpha", 1215.67, True, "lyman", r"Ly$\alpha$"),
    # Forbidden narrow lines
    Line("NII6548", 6549.86, False, "nii", "[N II] 6548"),
    Line("NII6584", 6585.27, False, "nii", "[N II] 6584"),
    Line("SII6716", 6718.29, False, "sii", "[S II] 6716"),
    Line("SII6731", 6732.67, False, "sii", "[S II] 6731"),
    Line("OIII4959", 4960.30, False, "oiii", "[O III] 4959"),
    Line("OIII5007", 5008.24, False, "oiii", "[O III] 5007"),
    Line("OIII4363", 4364.44, False, "oiii_aux", "[O III] 4363"),
    Line("OII3726", 3727.09, False, "oii", "[O II] 3726"),
    Line("OII3729", 3729.88, False, "oii", "[O II] 3729"),
    Line("NeIII3869", 3869.86, False, "neiii", "[Ne III] 3869"),
    Line("NeV3426", 3426.85, False, "nev", "[Ne V] 3426"),
    Line("OI6300", 6302.05, False, "oi", "[O I] 6300"),
    Line("SIII9069", 9071.10, False, "siii", "[S III] 9069"),
    Line("SIII9531", 9533.20, False, "siii", "[S III] 9531"),
)

BY_NAME: dict[str, Line] = {line.name: line for line in LINES}

#: Fixed atomic branching ratios.  Only ratios set by atomic physics belong
#: here; density-sensitive ratios such as [S II] 6716/6731 must stay free.
FIXED_RATIOS: dict[tuple[str, str], float] = {
    ("NII6584", "NII6548"): 1.0 / 2.94,  # [N II] 6548 / 6584
    ("OIII5007", "OIII4959"): 1.0 / 2.98,  # [O III] 4959 / 5007
    ("SIII9531", "SIII9069"): 1.0 / 2.47,  # [S III] 9069 / 9531
}


@dataclass(frozen=True)
class LineSystem:
    """A set of lines expected at one redshift, with their broad-line subset."""

    name: str
    members: tuple[str, ...]
    broad_members: tuple[str, ...] = ()
    notes: str = ""

    def lines(self) -> tuple[Line, ...]:
        return tuple(BY_NAME[n] for n in self.members)


#: Detection branches, chosen by what the red grism can actually see.
SYSTEMS: tuple[LineSystem, ...] = (
    LineSystem(
        "halpha_complex",
        ("Halpha", "NII6548", "NII6584", "SII6716", "SII6731", "OI6300"),
        ("Halpha",),
        "z ~ 0.90-1.82 in 12500-18500 A",
    ),
    LineSystem(
        "hbeta_oiii",
        ("Hbeta", "OIII4959", "OIII5007", "OIII4363", "Hgamma"),
        ("Hbeta", "Hgamma"),
        "z ~ 1.50-2.81",
    ),
    LineSystem(
        "paschen_beta",
        ("Pabeta", "SIII9531", "SIII9069"),
        ("Pabeta",),
        "z ~ 0-0.44; the lowest-redshift AGN probe in the red grism",
    ),
    LineSystem(
        "helium_paschen_gamma",
        ("HeI10830", "Pagamma", "Padelta"),
        ("HeI10830", "Pagamma"),
        "z ~ 0.14-0.71",
    ),
    LineSystem("oii_neiii", ("OII3726", "OII3729", "NeIII3869"), (), "z ~ 2.35-3.96"),
    LineSystem("mgii", ("MgII2796", "MgII2803"), ("MgII2796", "MgII2803"), "z ~ 3.47-5.61"),
)

BY_SYSTEM: dict[str, LineSystem] = {s.name: s for s in SYSTEMS}


def visible(
    line: Line | str,
    z: float,
    wmin: float = RGS_SCIENCE_WMIN_ANGSTROM,
    wmax: float = RGS_SCIENCE_WMAX_ANGSTROM,
) -> bool:
    """Is a line inside the usable wavelength interval at redshift ``z``?"""
    line = BY_NAME[line] if isinstance(line, str) else line
    return wmin <= line.rest * (1.0 + z) <= wmax


def redshift_range(
    line: Line | str,
    wmin: float = RGS_SCIENCE_WMIN_ANGSTROM,
    wmax: float = RGS_SCIENCE_WMAX_ANGSTROM,
) -> tuple[float, float]:
    """Redshift interval over which a line falls inside ``[wmin, wmax]``."""
    line = BY_NAME[line] if isinstance(line, str) else line
    return (wmin / line.rest - 1.0, wmax / line.rest - 1.0)


def visible_lines(
    z: float,
    wmin: float = RGS_SCIENCE_WMIN_ANGSTROM,
    wmax: float = RGS_SCIENCE_WMAX_ANGSTROM,
    names: tuple[str, ...] | None = None,
) -> tuple[Line, ...]:
    """All catalogue lines observable at redshift ``z``."""
    pool = LINES if names is None else tuple(BY_NAME[n] for n in names)
    return tuple(line for line in pool if visible(line, z, wmin, wmax))


def visible_systems(
    z: float,
    wmin: float = RGS_SCIENCE_WMIN_ANGSTROM,
    wmax: float = RGS_SCIENCE_WMAX_ANGSTROM,
    min_lines: int = 1,
    require_broad: bool = False,
) -> tuple[LineSystem, ...]:
    """Line systems with at least ``min_lines`` members observable at ``z``.

    With ``require_broad`` the system must also have a permitted member in
    range, which is the condition for the BLR branch to be testable at all.
    """
    out = []
    for system in SYSTEMS:
        seen = [ln for ln in system.lines() if visible(ln, z, wmin, wmax)]
        if len(seen) < min_lines:
            continue
        if require_broad and not any(
            ln.name in system.broad_members and ln.permitted for ln in seen
        ):
            continue
        out.append(system)
    return tuple(out)


@dataclass(frozen=True)
class CoverageReport:
    """What a given redshift hypothesis can and cannot test."""

    z: float
    visible: tuple[str, ...] = field(default_factory=tuple)
    systems: tuple[str, ...] = field(default_factory=tuple)
    broad_testable: tuple[str, ...] = field(default_factory=tuple)
    bpt_complete: bool = False


def coverage(
    z: float,
    wmin: float = RGS_SCIENCE_WMIN_ANGSTROM,
    wmax: float = RGS_SCIENCE_WMAX_ANGSTROM,
) -> CoverageReport:
    """Summarise line coverage at one redshift.

    ``bpt_complete`` marks the narrow interval where the classical [N II] BPT
    set (H-alpha, [N II] 6584, H-beta, [O III] 5007) is simultaneously visible.
    Outside it, BPT cannot be a selection gate.
    """
    seen = visible_lines(z, wmin, wmax)
    names = tuple(ln.name for ln in seen)
    systems = visible_systems(z, wmin, wmax)
    broad = tuple(ln.name for ln in seen if ln.permitted)
    bpt = all(n in names for n in ("Halpha", "NII6584", "Hbeta", "OIII5007"))
    return CoverageReport(
        z=z,
        visible=names,
        systems=tuple(s.name for s in systems),
        broad_testable=broad,
        bpt_complete=bpt,
    )
