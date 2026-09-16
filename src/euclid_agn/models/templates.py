"""Fixed-ratio emission templates for redshift *identification*.

Why this exists
---------------
Ranking hypotheses on the best fit with a free non-negative amplitude per line
cannot penalise a hypothesis for predicting a line that is not there: an absent
line simply gets amplitude zero.  A wrong identification that matches one real
feature with one of its components and mops up positive noise with the others
therefore scores as well as, and usually better than, the truth.  Measured on
659 real Q1 spectra, that ranking agreed with Euclid's SPE redshift 2.6 per cent
of the time, and only 19 per cent of the time when the true redshift was handed
to it as an explicit hypothesis.

Template redshift fitters avoid this by tying lines together: one amplitude per
template, ratios fixed at physically plausible values, so the template *must*
put flux where the spectrum has none if the identification is wrong.  That is
the evidence-against that free amplitudes lack.

The templates here are used **only** to choose the redshift.  Once it is
chosen, the measurement reverts to free amplitudes and the shared non-parametric
profile, exactly as before: nothing about the AGN measurement depends on the
ratios below.

Ratios are indicative, not calibrated.  Each system carries a few templates
spanning the range from star-forming to AGN-like excitation so that a real
object of either kind has one template within a factor of ~2 of its ratios;
the best template wins, and the identification statistic is the best of a
handful of one-parameter fits rather than one many-parameter fit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from euclid_agn.models.line_catalog import BY_NAME, BY_SYSTEM, LineSystem


@dataclass(frozen=True)
class EmissionTemplate:
    """Lines with fixed relative fluxes."""

    name: str
    system: str
    ratios: dict[str, float]  # line name -> flux relative to the reference line

    def visible(self, z: float, wmin: float, wmax: float) -> dict[str, float]:
        """Members inside the covered range at redshift ``z``."""
        return {
            name: ratio
            for name, ratio in self.ratios.items()
            if wmin <= BY_NAME[name].rest * (1.0 + z) <= wmax
        }


TEMPLATES: tuple[EmissionTemplate, ...] = (
    # --- H-alpha complex: star-forming through Seyfert-2 excitation ---------
    EmissionTemplate(
        "halpha_hii",
        "halpha_complex",
        {"Halpha": 1.0, "NII6584": 0.25, "NII6548": 0.085, "SII6716": 0.20,
         "SII6731": 0.15, "OI6300": 0.03},
    ),
    EmissionTemplate(
        "halpha_composite",
        "halpha_complex",
        {"Halpha": 1.0, "NII6584": 0.6, "NII6548": 0.2, "SII6716": 0.3,
         "SII6731": 0.22, "OI6300": 0.07},
    ),
    EmissionTemplate(
        "halpha_agn",
        "halpha_complex",
        {"Halpha": 1.0, "NII6584": 1.2, "NII6548": 0.41, "SII6716": 0.4,
         "SII6731": 0.35, "OI6300": 0.15},
    ),
    # --- H-beta / [O III]: from weak-[O III] star-forming to [O III]-dominated
    EmissionTemplate(
        "hbeta_hii",
        "hbeta_oiii",
        {"Hbeta": 1.0, "OIII5007": 1.5, "OIII4959": 0.5, "Hgamma": 0.47, "Hdelta": 0.26},
    ),
    EmissionTemplate(
        "hbeta_oiii_strong",
        "hbeta_oiii",
        {"Hbeta": 1.0, "OIII5007": 5.0, "OIII4959": 1.68, "Hgamma": 0.47, "Hdelta": 0.26},
    ),
    EmissionTemplate(
        "hbeta_agn",
        "hbeta_oiii",
        {"Hbeta": 1.0, "OIII5007": 10.0, "OIII4959": 3.36, "Hgamma": 0.47,
         "Hdelta": 0.26, "HeII4686": 0.25, "OIII4363": 0.2},
    ),
    # --- Paschen-beta with [S III] -------------------------------------------
    EmissionTemplate(
        "pabeta_hii",
        "paschen_beta",
        {"Pabeta": 1.0, "SIII9531": 1.5, "SIII9069": 0.6},
    ),
    EmissionTemplate(
        "pabeta_weak_siii",
        "paschen_beta",
        {"Pabeta": 1.0, "SIII9531": 0.4, "SIII9069": 0.16},
    ),
    # --- He I / Paschen-gamma / delta ----------------------------------------
    EmissionTemplate(
        "hei_paschen",
        "helium_paschen_gamma",
        {"HeI10830": 1.0, "Pagamma": 0.5, "Padelta": 0.3},
    ),
    EmissionTemplate(
        "paschen_only",
        "helium_paschen_gamma",
        {"HeI10830": 0.3, "Pagamma": 1.0, "Padelta": 0.55},
    ),
    # --- [O II] / [Ne III] ---------------------------------------------------
    EmissionTemplate(
        "oii_neiii",
        "oii_neiii",
        {"OII3726": 1.0, "OII3729": 1.3, "NeIII3869": 0.3},
    ),
    # --- Mg II doublet -------------------------------------------------------
    EmissionTemplate("mgii", "mgii", {"MgII2796": 1.0, "MgII2803": 0.8}),
    # --- C III] --------------------------------------------------------------
    EmissionTemplate("ciii", "ciii", {"CIII1909": 1.0}),
)

BY_SYSTEM_TEMPLATES: dict[str, tuple[EmissionTemplate, ...]] = {
    name: tuple(t for t in TEMPLATES if t.system == name) for name in BY_SYSTEM
}


def templates_for(system: LineSystem | str) -> tuple[EmissionTemplate, ...]:
    name = system if isinstance(system, str) else system.name
    return BY_SYSTEM_TEMPLATES.get(name, ())


def template_column(
    projected,
    template: EmissionTemplate,
    z: float,
    sigma_kms: float,
    min_containment: float = 0.8,
) -> tuple[np.ndarray | None, int]:
    """One column: the template's visible lines at their fixed ratios.

    Returns ``(column, n_lines)``; ``column`` is ``None`` if nothing is visible.
    Each line is a unit-flux LSF-broadened profile scaled by its ratio, so the
    single fitted amplitude is the flux of the reference line.

    A line whose profile is less than ``min_containment`` on covered pixels is
    left out: a template line on the very edge of the grid fits the edge
    artefact rather than the sky.  VERIFIED on a real Q1 spectrum where the
    winning template placed Pa-gamma on the red-edge spike at 18450 A.
    """
    from euclid_agn.spectra.lsf import effective_sigma, sigma_kms_to_angstrom

    wavelength = projected.wavelength
    visible = template.visible(z, float(wavelength[0]), float(wavelength[-1]))
    if not visible:
        return None, 0
    column = np.zeros(wavelength.size)
    n_used = 0
    for name, ratio in visible.items():
        centre = BY_NAME[name].rest * (1.0 + z)
        width = effective_sigma(sigma_kms_to_angstrom(sigma_kms, centre), projected.lsf_sigma)
        line = projected.line_column(centre, width)
        if min_containment > 0.0 and float(np.sum(line * projected.widths)) < min_containment:
            continue
        column += ratio * line
        n_used += 1
    if n_used == 0:
        return None, 0
    return column, n_used
