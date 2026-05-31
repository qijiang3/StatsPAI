# Monte Carlo CI Coverage Findings

This file records what the coverage suite is allowed to claim. The JSS
submission uses committed deep-audit artifacts in
`results_b1000/coverage_b1000.json` and
`results_b1000/coverage_robustness_b1000.json`; the regular pytest suite
keeps lower draw caps for wall-clock reasons.

## Headline B=1000 Audit

The canonical Track B audit materializes seven known-truth DGPs at
`B=1000`. The 99% Wilson band around nominal 0.95 is approximately
`[0.935, 0.967]`; rows above the band are treated as conservative
over-coverage, not as evidence of under-calibrated standard errors.

| Estimator | DGP | Coverage |
| --- | --- | --- |
| `sp.regress` (HC1) | RCT with covariates | 0.952 |
| `sp.regress` 2x2 DiD | 2-period homogeneous DiD | 0.955 |
| `sp.ivreg` (HC1) | Strong binary-Z IV | 0.962 |
| `sp.callaway_santanna` (REG, simple ATT) | Homogeneous staggered timing | 0.946 |
| `sp.ebalance` | CIA with 2 covariates | 1.000 |
| `sp.causal_question(design="dml")` | Binary-treatment IRM ATE | 0.969 |
| `sp.causal_question(design="causal_forest")` | AIPW-IF ATE DGP | 0.977 |

Interpretation:

- Closed-form OLS, DiD, IV, and simple Callaway-Sant'Anna rows sit inside
  the Wilson band.
- DML sits just above the upper edge; ebalance and causal forest are more
  visibly conservative. These are over-coverage findings, not hidden
  under-coverage.
- The expensive DML and causal-forest rows are no longer supported only by
  the lower-B pytest caps; the committed JSS artifacts record their
  explicit B=1000 rates.

## Resolved Finding

### `sp.callaway_santanna` simple-ATT aggregation

Previous finding: empirical 95% CI coverage was about 50% on a
homogeneous staggered DGP. Point estimates were unbiased, but simple-ATT
CIs were too tight.

Root causes fixed:

1. Group-time influence functions are estimated on the relevant
   treated/control subset, then embedded into the full unit universe for
   aggregation. They must be multiplied by `n_total / n_relevant` during
   that embedding.
2. The outcome-regression (`estimator="reg"`) influence function must
   include uncertainty from the control outcome regression. The previous
   implementation only carried the treated-side residual term.

Current result: the B=1000 deep audit reports `946/1000 = 0.946`, inside
the 99% Wilson band `[0.935, 0.967]`. The `04_csdid` R/Stata parity row
reports simple-ATT point-estimate parity at machine precision and
analytic-SE parity within the registered 1% tolerance.

## Robustness DGPs

The sibling robustness suite runs selected estimators under DGPs that
stress or violate identification assumptions. These rows are descriptive
failure-mode checks, not pass/fail nominal-calibration claims. Each row
has a documented band; a movement outside the band means the estimator
changed and the finding must be reviewed.

| Estimator | Stressor | Deep-audit coverage | Documented band |
| --- | --- | --- | --- |
| `sp.ivreg` (HC1) | Weak instrument: pi=0.10, median F = 2.51 | 0.882 (B=1000) | [0.85, 0.95] |
| `sp.callaway_santanna` (REG) | Heterogeneous timing and magnitude | 0.946 (B=1000) | [0.92, 0.96] |
| `sp.causal_question(causal_forest)` | Severe propensity-overlap loss | 0.983 (B=300) | [0.85, 0.99] |

Findings interpretation:

- Weak-IV under-coverage is textbook. With a first-stage F far below
  Stock-Yogo critical values, HC1 2SLS intervals miss truth more often
  than nominal 0.95. User-facing recovery routes are LIML and
  Anderson-Rubin inference, surfaced by the preflight/design-detect path.
- CS-DiD remains calibrated under the heterogeneous timing/magnitude DGP,
  consistent with cell-level ATT(g, t) estimation and simple-ATT
  aggregation.
- Causal forest under severe overlap loss over-covers. Extreme
  propensities inflate the AIPW influence-function variance and produce
  wide intervals; the audit treats this as a diagnostic warning, not as an
  efficiency claim.

## How to Run

Fast smoke:

```bash
pytest tests/coverage_monte_carlo/test_coverage.py::test_fast_ols_coverage_smoke
```

Canonical slow pytest suite, using the default lower draw caps:

```bash
pytest -m slow tests/coverage_monte_carlo/test_coverage.py
```

Robustness DGPs, also lower-capped for routine pytest use:

```bash
pytest -m slow tests/coverage_monte_carlo/test_coverage_robustness.py
```

Deep JSS audit artifacts:

```bash
python tests/coverage_monte_carlo/run_b1000.py
python tests/coverage_monte_carlo/run_robustness_b1000.py
```
