"""Physical constants and Euclid instrument facts used across the package.

Instrument numbers marked VERIFIED were measured from a real Q1 SIR product
(tile 102160339) on 2026-09-16; see JOURNAL.md for the measurement.
"""

from __future__ import annotations

# Exact by definition of the metre and the second.
C_KMS: float = 299792.458

# --- Euclid NISP red grism (RGS), Q1 SIR combined spectra ------------------
# VERIFIED from EUC_SIR_W-COMBSPEC_102160339_2024-11-05T16:26:34.614296Z.fits
SIR_WMIN_ANGSTROM: float = 11900.0  # WMIN header keyword
SIR_BINWIDTH_ANGSTROM: float = 13.4  # BINWIDTH header keyword
SIR_BINCOUNT: int = 531  # BINCOUNT header keyword
SIR_WMAX_ANGSTROM: float = SIR_WMIN_ANGSTROM + SIR_BINWIDTH_ANGSTROM * (SIR_BINCOUNT - 1)

# Science-usable red-grism interval quoted by the Q1 release notes.  The stored
# grid is wider than this (11900-19002 A) and the edges are noisy.
RGS_SCIENCE_WMIN_ANGSTROM: float = 12500.0
RGS_SCIENCE_WMAX_ANGSTROM: float = 18500.0

# The stored sampling is ~1 LSF sigma per bin for compact sources, so model
# fluxes must be integrated across a pixel rather than sampled at its centre.
SIR_FSCALE_KEYWORD: str = "FSCALE"

FLUX_UNIT: str = "erg / (s cm2 Angstrom)"
VARIANCE_UNIT: str = "erg2 / (s2 cm4 Angstrom2)"
