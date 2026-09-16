"""Archive table names, column selections and field footprints.

Table and column names below were read from the live IRSA TAP_SCHEMA on
2026-09-16 (VERIFIED), not copied from documentation.
"""

from __future__ import annotations

from dataclasses import dataclass

# --- IRSA Q1 table names ---------------------------------------------------
MER_CATALOGUE = "euclid_q1_mer_catalogue"
MER_MORPHOLOGY = "euclid_q1_mer_morphology"
SPECTRA_ASSOCIATION = "euclid.objectid_spectrafile_association_q1"
PHZ_PHOTO_Z = "euclid_q1_phz_photo_z"
SPE_CLASSIFICATION = "euclid_q1_spectro_zcatalog_spe_classification"
SPE_GALAXY_CANDIDATES = "euclid_q1_spectro_zcatalog_spe_galaxy_candidates"
SPE_QSO_CANDIDATES = "euclid_q1_spectro_zcatalog_spe_qso_candidates"
SPE_STAR_CANDIDATES = "euclid_q1_spectro_zcatalog_spe_star_candidates"
SPE_QUALITY = "euclid_q1_spectro_zcatalog_spe_quality"
SPE_LINE_FEATURES = "euclid_q1_spe_lines_line_features"

#: Public IRSA S3 mirror of the Q1 release.
IRSA_S3_BUCKET = "nasa-irsa-euclid-q1"
IRSA_S3_ROOT = f"{IRSA_S3_BUCKET}/q1"
IRSA_TAP_SYNC = "https://irsa.ipac.caltech.edu/TAP/sync"
IRSA_TAP_ASYNC = "https://irsa.ipac.caltech.edu/TAP/async"

#: MER columns kept in the manifest.  Photometry and morphology are carried as
#: *context*: they are reported in the catalogue and used to stratify the
#: selection function, never to select AGN candidates.
MER_MANIFEST_COLUMNS: tuple[str, ...] = (
    "object_id",
    "ra",
    "dec",
    "tileid",
    "flux_h_2fwhm_aper",
    "fluxerr_h_2fwhm_aper",
    "flux_y_2fwhm_aper",
    "flux_j_2fwhm_aper",
    "flux_vis_2fwhm_aper",
    "flux_detection_total",
    "segmentation_area",
    "semimajor_axis",
    "ellipticity",
    "position_angle",
    "kron_radius",
    "fwhm",
    "mu_max",
    "point_like_flag",
    "point_like_prob",
    "extended_flag",
    "extended_prob",
    "spurious_flag",
    "det_quality_flag",
    "deblended_flag",
    "blended_prob",
    "variable_flag",
    "gal_ebv",
    "has_spectrum",
)

MORPHOLOGY_MANIFEST_COLUMNS: tuple[str, ...] = (
    "sersic_sersic_nir_radius",
    "sersic_sersic_nir_axis_ratio",
    "sersic_sersic_nir_index",
    "sersic_sersic_vis_radius",
    "sersic_sersic_vis_axis_ratio",
    "sersic_sersic_vis_index",
    "sersic_angle",
    "concentration",
    "asymmetry",
    "smoothness",
    "gini",
    "moment_20",
)

PHZ_MANIFEST_COLUMNS: tuple[str, ...] = (
    "phz_median",
    "phz_mode_1",
    "phz_mode_1_area",
    "phz_mode_2",
    "phz_mode_2_area",
    "phz_classification",
    "phz_flags",
)

SPE_CLASSIFICATION_COLUMNS: tuple[str, ...] = (
    "spe_class",
    "spe_star_prob",
    "spe_gal_prob",
    "spe_qso_prob",
)

SPE_QUALITY_COLUMNS: tuple[str, ...] = (
    "spe_grism",
    "spe_w_min",
    "spe_w_max",
    "spe_npix",
    "spe_n_dith_max",
    "spe_n_dith_med",
    "spe_error_flag",
    "spe_warning_flag",
)


CAOM_TILE_ASSOCIATION = "euclid.tileid_association_q1"
CAOM_PLANE = "euclid.plane_euclid_q1"


@dataclass(frozen=True)
class FieldFootprint:
    """Cone enclosing a Q1 field, used to resolve a field name to its tiles.

    VERIFIED on 2026-09-16: these four cones select 124 + 148 + 72 + 8 = 352
    distinct tiles, exactly the number of distinct tiles in
    ``euclid.tileid_association_q1``.  The cones therefore partition Q1 with no
    overlap and no omission.  They are selectors, not a survey-area definition.

    The centres were checked against MER source counts inside a 0.2 deg cone
    (EDF-S 57074, EDF-F 47253, LDN1641 9673 sources).
    """

    name: str
    ra: float
    dec: float
    radius_deg: float

    def cone_adql(self, point_expr: str = "p.pt") -> str:
        return f"CONTAINS({point_expr}, CIRCLE('ICRS', {self.ra}, {self.dec}, {self.radius_deg})) = 1"


FIELDS: dict[str, FieldFootprint] = {
    "EDF-N": FieldFootprint("EDF-N", 269.73, 66.018, 6.0),
    "EDF-S": FieldFootprint("EDF-S", 61.241, -48.423, 6.0),
    "EDF-F": FieldFootprint("EDF-F", 52.932, -28.088, 5.0),
    "LDN1641": FieldFootprint("LDN1641", 85.4, -8.0, 4.0),
}
