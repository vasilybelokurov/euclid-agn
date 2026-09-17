"""Spectral template libraries for the redshift engine.

Three classes of object, three sources, one interface:

``GALAXY``  XSL simple stellar populations (Verro et al. 2022): empirical
            stellar spectra, 350-2475 nm, so the H-band features - the 1.6 micron
            bump, CO bandheads, Mg I / Al I, Paschen absorption - are real, not
            synthetic.  Reduced to a low-dimensional PCA basis so the continuum
            of any galaxy is a few linear coefficients.
``QSO``     Glikman, Helfand & White (2006) optical-to-infrared quasar composite,
            0.58-3.5 micron rest, for z < ~1.5; broad lines are carried by the
            composite's own line profiles plus the pipeline's broad components.
``STAR``    XSL DR3 stellar spectra (loaded on demand).

Ground-based libraries carry telluric gaps.  Measured on the XSL SSPs
(JOURNAL, 2026-09-17): rest 1.35-1.425 micron and > 1.80 micron have fractional
roughness 9-25 per cent against 1-2 per cent elsewhere.  Euclid observes those
regions cleanly, so a template must not carry the gap into the fit: gap pixels
are masked and bridged with a smooth low-order fit through the clean neighbours.
The bridge is a model, and the mask that produced it is recorded.

Every template is projected onto a spectrum's own grid at a trial redshift the
same way the line models are - LSF-broadened, integrated across each kept
pixel's true edges - so template and line columns are directly comparable in
one linear solve.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

from euclid_agn.constants import C_KMS
from euclid_agn.numerics import blas_safe

log = logging.getLogger(__name__)

DEFAULT_ROOT = Path("~/data/euclid/templates").expanduser()

#: Rest-frame ranges (Angstrom) unusable in ground-based NIR libraries.
XSL_TELLURIC_GAPS: tuple[tuple[float, float], ...] = ((13500.0, 14250.0), (18000.0, 25000.0))

_XSL_NAME = re.compile(r"XSL_SSP_logT(?P<logt>[\d.]+)_MH(?P<mh>-?[\d.]+)_(?P<imf>\w+)_(?P<iso>\w+)\.fits")


@dataclass(frozen=True)
class Template:
    """One rest-frame template on its own wavelength grid (Angstrom, F_lambda)."""

    name: str
    kind: str  # GALAXY | QSO | STAR
    wavelength: np.ndarray
    flux: np.ndarray
    mask: np.ndarray  # True where the template is trustworthy
    metadata: dict = field(default_factory=dict)

    def bridged(self) -> Template:
        """Replace masked stretches by a smooth low-order interpolation.

        A cubic in log-wavelength through the clean pixels within a window of
        each gap: enough to carry the continuum shape across, deliberately too
        stiff to invent features.
        """
        flux = self.flux.copy()
        bad = ~self.mask
        if not bad.any():
            return self
        edges = np.flatnonzero(np.diff(np.concatenate([[0], bad.astype(int), [0]])))
        for start, stop in zip(edges[::2], edges[1::2], strict=True):
            lo, hi = self.wavelength[start], self.wavelength[stop - 1]
            width = hi - lo
            window = (self.wavelength > lo - 1.5 * width) & (self.wavelength < hi + 1.5 * width) & self.mask
            if window.sum() < 20:
                continue
            x = np.log(self.wavelength[window]); y = np.log(np.clip(self.flux[window], 1e-300, None))
            coefficients = np.polyfit(x - x.mean(), y, 3)
            xg = np.log(self.wavelength[start:stop]) - x.mean()
            flux[start:stop] = np.exp(np.polyval(coefficients, xg))
        return Template(self.name, self.kind, self.wavelength, flux, np.ones_like(self.mask), {**self.metadata, "bridged_gaps": True})


def _log_grid_from_header(header, n: int, unit_to_angstrom: float) -> np.ndarray:
    crpix = header.get("CRPIX1", 1)
    return 10 ** ((np.arange(n) + 1 - crpix) * header["CDELT1"] + header["CRVAL1"]) * unit_to_angstrom


def load_xsl_ssp(path: str | Path, gaps=XSL_TELLURIC_GAPS) -> Template:
    """One XSL SSP FITS file -> Template (rest Angstrom, telluric gaps masked)."""
    from astropy.io import fits

    path = Path(path)
    with fits.open(path) as handle:
        header = handle[0].header
        flux = np.asarray(handle[0].data, dtype=np.float64)
    unit = str(header.get("CUNIT1", "nm")).strip().lower()
    wavelength = _log_grid_from_header(header, flux.size, 10.0 if unit == "nm" else 1.0)
    mask = np.ones(flux.size, dtype=bool)
    for lo, hi in gaps:
        mask &= ~((wavelength >= lo) & (wavelength <= hi))
    mask &= np.isfinite(flux) & (flux > 0)
    match = _XSL_NAME.match(path.name)
    metadata = {"source": str(path), "library": "XSL-SSP"}
    if match:
        metadata.update({"log_age": float(match["logt"]), "mh": float(match["mh"]), "imf": match["imf"], "isochrone": match["iso"]})
    return Template(path.stem, "GALAXY", wavelength, flux, mask, metadata)


def load_xsl_ssp_library(root: str | Path = DEFAULT_ROOT, pattern: str = "xsl_ssp/**/*.fits",
                         log_age_min: float = 7.7, mh_min: float = -1.0) -> list[Template]:
    """All XSL SSPs under ``root`` matching the age/metallicity selection.

    The default drops [M/H] < -1: such metal-poor populations do not dominate
    the light of H < 22.5 galaxies and only add degrees of freedom.
    """
    files = sorted(Path(root).expanduser().glob(pattern))
    out = []
    for f in files:
        match = _XSL_NAME.match(f.name)
        if match and (float(match["logt"]) < log_age_min or float(match["mh"]) < mh_min):
            continue
        out.append(load_xsl_ssp(f))
    if not out:
        raise FileNotFoundError(f"no XSL SSP files under {root}/{pattern}")
    return out


def load_glikman_composite(path: str | Path = DEFAULT_ROOT / "qso" / "table7.dat", geometric: bool = False) -> Template:
    """Glikman et al. 2006 optical-to-infrared quasar composite (VizieR J/ApJ/640/579, table7)."""
    # whitespace-separated: wavelength [A], arithmetic mean, error, geometric mean
    values = np.loadtxt(Path(path).expanduser())
    wavelength = values[:, 0]
    flux = values[:, 3] if (geometric and values.shape[1] > 3) else values[:, 1]
    order = np.argsort(wavelength)
    wavelength, flux = wavelength[order], flux[order]
    mask = np.isfinite(flux) & (flux > 0)
    return Template("glikman2006_composite", "QSO", wavelength, flux, mask,
                    {"source": str(path), "library": "Glikman+2006", "mean": "geometric" if geometric else "arithmetic"})


# --- projection onto a Euclid spectrum -------------------------------------------


@blas_safe
def project_template(template: Template, projected, z: float, extra_sigma_kms: float = 0.0) -> np.ndarray | None:
    """Template at redshift ``z`` on the spectrum's kept pixels.

    The rest-frame template is redshifted, smoothed to the object's LSF (the
    template's own resolution is far higher than R ~ 450 and is ignored), and
    integrated across each kept pixel's true edges - the same treatment the line
    models receive.  Returns ``None`` if the template does not cover the kept
    range at this redshift.  Normalised to unit RMS over the kept pixels (PCA
    components are mean-centred, so a mean normalisation would divide by zero)
    so fitted coefficients are comparable flux scales.
    """
    observed = template.wavelength * (1.0 + z)
    lo, hi = projected.edges[:, 0].min(), projected.edges[:, 1].max()
    if observed[0] > lo or observed[-1] < hi:
        return None
    flux = template.bridged().flux if not template.mask.all() else template.flux
    # smooth to the LSF (plus any intrinsic broadening) in the observed frame
    dlog = np.median(np.diff(np.log(observed)))
    sigma_log = np.sqrt((projected.lsf_sigma / np.median(projected.wavelength)) ** 2 + (extra_sigma_kms / C_KMS) ** 2)
    smoothed = gaussian_filter1d(flux, max(sigma_log / dlog, 0.5), mode="nearest")
    # integrate across kept pixel edges via the cumulative integral
    cumulative = np.concatenate([[0.0], np.cumsum(0.5 * (smoothed[1:] + smoothed[:-1]) * np.diff(observed))])
    lower = np.interp(projected.edges[:, 0], observed, cumulative)
    upper = np.interp(projected.edges[:, 1], observed, cumulative)
    column = (upper - lower) / projected.widths
    scale = float(np.sqrt(np.mean(column**2)))
    return column / scale if np.isfinite(scale) and scale > 0 else None


# --- low-dimensional continuum basis --------------------------------------------


@dataclass(frozen=True)
class ContinuumBasis:
    """PCA basis of a template family on a common rest-frame log grid."""

    kind: str
    wavelength: np.ndarray
    mean: np.ndarray
    components: np.ndarray  # (n_components, n_wavelength)
    explained: np.ndarray
    metadata: dict = field(default_factory=dict)

    @property
    def n_components(self) -> int:
        return self.components.shape[0]

    def templates(self) -> list[Template]:
        """The mean and each component as Templates (components can be negative)."""
        mask = np.ones(self.wavelength.size, dtype=bool)
        out = [Template(f"{self.kind}_mean", self.kind, self.wavelength, self.mean, mask, {"basis": "mean"})]
        for i, comp in enumerate(self.components):
            out.append(Template(f"{self.kind}_pc{i + 1}", self.kind, self.wavelength, comp, mask, {"basis": f"pc{i + 1}"}))
        return out


def build_pca_basis(templates: list[Template], n_components: int = 5, wmin: float = 6000.0,
                    wmax: float = 19000.0, dlog: float = 1e-4) -> ContinuumBasis:
    """PCA of bridged templates on a common log grid, normalised to unit mean.

    ``wmin``-``wmax`` (rest Angstrom) is the range Euclid can see for the class
    over the redshifts scanned: 12500/(1+1.0) = 6250 A at the blue end for
    z <= 1, 18500 A at the red end for z = 0.  Restricting the PCA to it keeps
    the components describing what the grism sees.  (First version used 8000 A
    and silently returned None for every z > 0.56.)
    """
    grid = np.exp(np.arange(np.log(wmin), np.log(wmax), dlog))
    rows = []
    for t in templates:
        b = t.bridged()
        f = np.interp(grid, b.wavelength, b.flux)
        rows.append(f / np.mean(f))
    matrix = np.vstack(rows)
    mean = matrix.mean(axis=0)
    centred = matrix - mean
    u, s, vt = np.linalg.svd(centred, full_matrices=False)
    explained = s**2 / np.sum(s**2)
    return ContinuumBasis(templates[0].kind, grid, mean, vt[:n_components], explained[:n_components],
                          {"n_templates": len(templates), "wmin": wmin, "wmax": wmax})


@lru_cache(maxsize=4)
def galaxy_basis(root: str = str(DEFAULT_ROOT), n_components: int = 5) -> ContinuumBasis:
    return build_pca_basis(load_xsl_ssp_library(root), n_components=n_components)
