"""Host-unbiased AGN search in Euclid NISP slitless spectra.

Canonical internal units used throughout the package:

======================  ================================
quantity                unit
======================  ================================
wavelength              Angstrom
flux density            erg s^-1 cm^-2 Angstrom^-1
variance                (erg s^-1 cm^-2 Angstrom^-1)^2
velocity                km s^-1
======================  ================================

Conversion from the archive representation to these units happens exactly once,
in :mod:`euclid_agn.io.sir`.  Everything downstream assumes canonical units.
"""

__version__ = "0.1.0"

from euclid_agn.constants import C_KMS  # noqa: F401

__all__ = ["C_KMS", "__version__"]
