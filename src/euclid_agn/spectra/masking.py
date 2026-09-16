"""SIR mask-bit handling.

The mask bit definitions are carried in the primary header of every SIR
combined-spectra file as ``HIERARCH MSK_FLAG_*`` keywords.  VERIFIED values from
tile 102160339::

    MSK_FLAG_GOOD     = 0
    MSK_FLAG_NOT_USE  = 1
    MSK_FLAG_LOW_SNR  = 2
    MSK_FLAG_EXT_PRB  = 4
    MSK_FLAG_HIGH     = 8
    MSK_FLAG_LOW      = 16
    MSK_FLAG_REL_FLUX = 32
    MSK_FLAG_ABS_FLUX = 64

Despite the "Mask bit position" comment in the header the values behave as bit
*masks*: observed MASK column values in real data are combinations such as
65 = 64 | 1 and 67 = 64 | 2 | 1.

The defaults below reject only ``NOT_USE``.  ``LOW_SNR`` is deliberately *not*
rejected: it is set on ~32 per cent of pixels and carries information that the
likelihood already down-weights through VAR.  ``ABS_FLUX``/``REL_FLUX`` flag
absolute/relative flux-calibration concerns and are recorded, not discarded.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Canonical fallback definitions, used when a file does not carry the keywords.
MASK_BITS: dict[str, int] = {
    "GOOD": 0,
    "NOT_USE": 1,
    "LOW_SNR": 2,
    "EXT_PRB": 4,
    "HIGH": 8,
    "LOW": 16,
    "REL_FLUX": 32,
    "ABS_FLUX": 64,
}

#: Bits that make a pixel unusable for fitting by default.
DEFAULT_REJECT_BITS: tuple[str, ...] = ("NOT_USE",)


@dataclass(frozen=True)
class MaskDefinition:
    """Mask-bit definition attached to a particular SIR file."""

    bits: dict[str, int]

    @classmethod
    def default(cls) -> MaskDefinition:
        return cls(bits=dict(MASK_BITS))

    @classmethod
    def from_header(cls, header) -> MaskDefinition:
        """Read ``MSK_FLAG_*`` keywords from a FITS header, falling back to defaults."""
        bits = {}
        for key in header:
            if key.startswith("MSK_FLAG_"):
                bits[key[len("MSK_FLAG_") :]] = int(header[key])
        if not bits:
            return cls.default()
        return cls(bits=bits)

    def value(self, name: str) -> int:
        try:
            return self.bits[name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise KeyError(f"unknown mask bit {name!r}; known: {sorted(self.bits)}") from exc

    def reject_value(self, names: tuple[str, ...] = DEFAULT_REJECT_BITS) -> int:
        """Bitwise OR of the named bits."""
        out = 0
        for name in names:
            out |= self.value(name)
        return out


def bit_is_set(mask: np.ndarray, bit: int) -> np.ndarray:
    """Boolean array: is ``bit`` set in each element of ``mask``?

    ``bit == 0`` means "GOOD" and is interpreted as ``mask == 0``.
    """
    mask = np.asarray(mask, dtype=np.int64)
    if bit == 0:
        return mask == 0
    return (mask & bit) > 0


def usable_pixels(
    mask: np.ndarray,
    definition: MaskDefinition | None = None,
    reject: tuple[str, ...] = DEFAULT_REJECT_BITS,
) -> np.ndarray:
    """Boolean array marking pixels that may enter a fit."""
    definition = definition or MaskDefinition.default()
    reject_value = definition.reject_value(reject)
    mask = np.asarray(mask, dtype=np.int64)
    if reject_value == 0:
        return np.ones(mask.shape, dtype=bool)
    return (mask & reject_value) == 0


def mask_bit_fractions(
    mask: np.ndarray, definition: MaskDefinition | None = None
) -> dict[str, float]:
    """Fraction of pixels with each known bit set (diagnostic, never a cut)."""
    definition = definition or MaskDefinition.default()
    mask = np.asarray(mask, dtype=np.int64)
    n = mask.size
    if n == 0:
        return {name: float("nan") for name in definition.bits}
    return {
        name: float(np.count_nonzero(bit_is_set(mask, bit))) / n
        for name, bit in definition.bits.items()
    }
