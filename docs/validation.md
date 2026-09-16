# Validation strategy

Nothing may be concluded about AGN detection performance before these exist.

1. **Synthetic unit-level spectra.** `euclid_agn.validation.simulator` generates
   continuum, arbitrary NLR profile, broad components, the real Euclid grid,
   LSF convolution, heteroscedastic variance, masks, multiple dithers and
   contamination-like artefacts, deterministically from a seed. Every fitting
   primitive passes synthetic tests before touching real data.

2. **Injection into real Euclid spectra.** The primary completeness experiment:
   inject line systems into real control spectra and refit, varying redshift,
   broad flux and width, NLR strength, continuum, effective LSF, source size,
   spectral quality, mask fraction and contamination. The product is
   `P(detected | injected parameters, host and data properties)`, not a single
   flux-completeness curve.

3. **Empirical nulls.** Real control spectra, off-line redshift hypotheses,
   dither-incoherent artefacts, contamination-rich controls, and broad
   components attached to forbidden transitions (which the machinery can fit but
   the AGN model must never interpret as BLR emission).

4. **External spectroscopy.** EDF-N, where DESI provides independent broad- and
   narrow-line AGN labels. Partitioned into development, validation and a final
   blind holdout; thresholds are fixed before the holdout is opened. Labels are
   used for evaluation, never as input filters.

Performance is reported stratified by redshift, H_E, spectral S/N, effective
LSF, source size, axis ratio, stellar mass, SFR, morphological class,
contamination state and AGN class. That stratification is the test of the
"all hosts" claim.
