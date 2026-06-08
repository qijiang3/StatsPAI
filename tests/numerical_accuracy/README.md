# Numerical-accuracy certification

Tests here certify StatsPAI's numerical kernels against **published,
multiple-precision *certified* reference values** — not against R/Stata output
(`tests/reference_parity/`) or against recovered population parameters.

## Why this is a separate directory

The JSS manuscript headlines a fixed count of `tests/reference_parity/` tests as
the cross-language / DGP-recovery parity evidence, and a count-drift guard
(`tests/test_jss_validation_api.py`) pins it. NIST StRD certification is a
*different* kind of evidence (certified-value numerical accuracy), so it lives
here to keep that headline count meaningful and stable.

## Current suites

- **`test_nist_strd_ols.py`** — NIST/ITL Statistical Reference Datasets (StRD)
  Linear Least Squares regression. All 11 datasets (Norris, Pontius, NoInt1/2,
  Filip, Longley, Wampler1–5) are bundled verbatim under
  `_fixtures/nist_strd/` (see that directory's `PROVENANCE.md`). The tests read
  the embedded certified coefficients / standard errors and check the
  `ols_fit` kernel and `OLSEstimator` covariance against them, plus a
  QR-beats-normal-equations guard on the ill-conditioned designs.
