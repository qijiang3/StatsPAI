# `sp.dml` and the DoubleML reference implementation

`sp.dml` is StatsPAI's implementation of the Double/Debiased Machine
Learning (DML) framework of Chernozhukov et al. (2018). The canonical
reference implementations are the [DoubleML][doubleml] R package
(Bach, Kurz, Chernozhukov, Spindler & Klaassen, JSS 108(3), 2024)
and the [doubleml-for-py][doubleml-py] Python package
(Bach, Chernozhukov, Kurz & Spindler, JMLR 23(53), 2022).

This guide covers (1) how the `sp.dml` API maps onto DoubleML, (2)
where the numbers agree, and (3) what the differences are and why
they exist.

[doubleml]: https://github.com/DoubleML/doubleml-for-r
[doubleml-py]: https://github.com/DoubleML/doubleml-for-py

## API mapping

`sp.dml` is a single dispatcher; the `model=` argument selects the
estimator and matches one of DoubleML's classes one-to-one.

| Model | `sp.dml(...)` | `doubleml.DoubleML*(...)` | DoubleML R |
| --- | --- | --- | --- |
| Partially linear regression | `sp.dml(model='plr', ml_g=..., ml_m=...)` | `DoubleMLPLR(ml_l=..., ml_m=...)` | `DoubleMLPLR$new(...)` |
| Interactive regression (AIPW ATE) | `sp.dml(model='irm', ml_g=..., ml_m=...)` | `DoubleMLIRM(ml_g=..., ml_m=...)` | `DoubleMLIRM$new(...)` |
| Partially linear IV | `sp.dml(model='pliv', instrument=..., ml_g=..., ml_m=..., ml_r=...)` | `DoubleMLPLIV(ml_l=..., ml_m=..., ml_r=...)` | `DoubleMLPLIV$new(...)` |
| Interactive IV (LATE) | `sp.dml(model='iivm', instrument=..., ml_g=..., ml_m=..., ml_r=...)` | `DoubleMLIIVM(ml_g=..., ml_m=..., ml_r=...)` | `DoubleMLIIVM$new(...)` |

Anything that takes a scikit-learn estimator on the DoubleML side also
works in `sp.dml`. StatsPAI additionally accepts string shortcuts
(`'rf'`, `'gbm'`, `'lasso'`, `'ridge'`, `'linear'`, `'logistic'`,
`'xgb'`, `'lgbm'`) for the common nuisance learners.

## Same-DGP, same-seed numerical agreement

The fixture in `tests/reference_parity/_fixtures/dml_data.csv` is a
seed-42 DGP with `n=1000`, `p=10`, true treatment effect `θ=0.5`. The
external parity test
[`tests/external_parity/test_dml_python_parity.py`](https://github.com/brycewang-stanford/StatsPAI/blob/main/tests/external_parity/test_dml_python_parity.py)
runs `sp.dml` and `doubleml-for-py` on this fixture with identical
scikit-learn learners (`LassoCV(cv=5)` for regression,
`LogisticRegressionCV(cv=5)` for binary propensity) under a fixed
seed.

The non-instrumented models (PLR, IRM) use `dml_data.csv`; the
instrumented models (PLIV, IIVM) use the companion `dml_iv_data.csv`
(n=2000, with a continuous instrument `z_c` and a binary instrument
`z_b`; see `_generate_dml_iv_data.py`). All four DoubleML model classes
are pinned against `doubleml-for-py`.

| Model | `sp.dml` (StatsPAI 1.16.1) | `doubleml-for-py` 0.11.3 | `DoubleML` R 1.0.2 (cv.glmnet) |
| --- | --- | --- | --- |
| **PLR** (continuous d) | **+0.5590 ± 0.0331** | **+0.5590 ± 0.0331** | +0.5368 ± 0.0335 |
| **IRM** (binary d, AIPW) | -0.0191 ± 0.0766 | -0.0267 ± 0.0742 | +0.0066 ± 0.0744 |
| **PLIV** (continuous d, instrument `z_c`) | **+0.5117 ± 0.0195** | **+0.5117 ± 0.0195** | — (not pinned on R side) |
| **IIVM** (binary d, instrument `z_b`, LATE) | +0.5495 ± 0.0924 | +0.5618 ± 0.0919 | — (not pinned on R side) |

- **PLR**: `sp.dml` and `doubleml-for-py` agree to **machine precision**
  on both the point estimate and the standard error — |Δ| = 1.1 × 10⁻¹⁶
  on the coefficient and 1.4 × 10⁻¹⁷ on the standard error, i.e. one
  float64 unit in the last place. This is exact numerical equivalence
  under shared scikit-learn folds — both implementations evaluate the
  same Neyman-orthogonal score on the same CV-fold partition under a
  fixed seed. The slight deviation from the R reference (~4.1%) reflects
  glmnet's penalty path differing fractionally from scikit-learn's
  `LassoCV`; the R reference is pinned by
  [`tests/reference_parity/test_dml_parity.py`](https://github.com/brycewang-stanford/StatsPAI/blob/main/tests/reference_parity/test_dml_parity.py)
  to within 7% relative.

- **IRM**: All three implementations land statistically at zero (the
  truth for this DGP). `sp.dml` and `doubleml-for-py` differ by
  0.0076 absolute — about one-tenth of a standard error — owing to
  internal differences in how the two AIPW scores are constructed. This
  residual is verified *not* to come from propensity trimming (matching
  the clip thresholds leaves it unchanged) nor from IPW normalization
  (toggling `doubleml-for-py`'s `normalize_ipw` leaves it unchanged). The
  external parity test tolerates 0.05 absolute deviation, which is
  roughly two-thirds of one SE on this fixture.

- **PLIV**: Like PLR, the partially linear IV estimator residualises
  `y`, `d`, and the instrument `z_c` on `X` and evaluates the same
  partialling-out score on a shared `KFold` partition. `sp.dml` and
  `doubleml-for-py` agree to **machine precision** on both the
  coefficient (|Δ| = 0) and the standard error (|Δ| ~ 3 × 10⁻¹⁷).

- **IIVM**: The interactive-IV LATE estimator behaves like IRM — its
  AIPW-style score leaves fold-conditional construction details
  unspecified, so `sp.dml` and `doubleml-for-py` agree to ~1.2 × 10⁻²
  (≈ 0.13 SE) rather than to machine precision. Both land near the true
  LATE of 0.5 (0.549 vs 0.562). The external parity test tolerates 0.05
  absolute, matching the IRM discipline.

## When to expect divergence

`sp.dml` deviates from `doubleml-for-py` only in implementation
details that the original Chernozhukov et al. (2018) score leaves
unspecified:

1. **Propensity trimming**: `sp.dml` clips propensities to
   `[0.01, 0.99]`; `doubleml-for-py` applies no clip by default. On this
   fixture few propensities approach the boundary, so the clip is *not*
   what drives the small IRM gap — matching the thresholds (or removing
   the clip) leaves both estimates unchanged. Trimming matters only when
   the estimated propensity has mass near 0 or 1, where the AIPW score is
   numerically unstable.
2. **Repeated cross-fitting aggregation**: `n_rep > 1` aggregates by
   median in both. With `n_rep=1` the seed fully determines folds.
3. **Convenience defaults**: `sp.dml`'s string aliases
   (`ml_g='rf'`, etc.) map to specific scikit-learn configurations
   (e.g. `RandomForestRegressor(n_estimators=200)`). Passing an
   explicit sklearn estimator removes this layer.

For audit-grade numerical equivalence, supply the same
`sklearn`-compatible estimators to both libraries (as the external
parity test does): the partialling-out models (PLR, PLIV) then agree
with `doubleml-for-py` to machine precision under a fixed seed
(verified above), and the AIPW models (IRM, IIVM) agree up to the small
score-construction difference noted above (≈ 0.10–0.13 SE). All four
DoubleML model classes are pinned numerically against `doubleml-for-py`
in `tests/external_parity/test_dml_python_parity.py`.

## Estimation procedure and nuisance learners

**DML2, not DML1.** `sp.dml` solves the *pooled* moment equation across
all cross-fitting folds at once — i.e. the **DML2** estimator of
Chernozhukov et al. (2018, Def. 3.2), which is also DoubleML's default
`dml_procedure`. For the partially-linear models this is the closed-form
`theta = sum(d_tilde * y_tilde) / sum(d_tilde**2)` over *all* out-of-fold
residuals (not a per-fold DML1 average). This pooled identity, the
solved Neyman moment, and the sandwich-variance formula are checked
directly in `tests/test_dml_orthogonality_invariants.py`. The per-fold
DML1 procedure is not currently exposed; for well-sized folds the two
agree closely and DML2 is the recommended default.

**Nuisance learners.** Any scikit-learn-compatible estimator can be
passed to `ml_g` / `ml_m` / `ml_r`, exactly as in DoubleML; string
aliases (`'lasso'`, `'rf'`, `'gbm'`, …) are convenience shortcuts for
common configurations. `sp.dml` does **not** ship a theory-driven
"rigorous"/plug-in lasso (the `hdm` `rlasso` of Belloni–Chernozhukov–
Hansen): that estimator is specific to the R `hdm` package, and
`doubleml-for-py` likewise relies on scikit-learn learners rather than
bundling it. The scikit-learn cross-validated `LassoCV` is the
Python-ecosystem analogue for a sparse linear nuisance, and a user who
wants plug-in penalty selection can pass any custom estimator that
implements the scikit-learn `fit`/`predict` API.

## Running the parity tests yourself

```bash
pip install -e ".[dev,parity]"   # the parity extra adds doubleml-for-py
                                  # (not a runtime dependency of StatsPAI)

# Python-side parity (sp.dml vs doubleml-for-py) — runs to machine precision.
# Without the parity extra this test skips cleanly instead of failing.
pytest tests/external_parity/test_dml_python_parity.py -v

# R-side parity (requires R + DoubleML + mlr3 installed locally)
pytest tests/reference_parity/test_dml_parity.py -v
```

The R-side fixture (`dml_R.json`) was generated once on R 4.5.2 with
`DoubleML` 1.0.2 + `mlr3learners` 0.14.0 + `cv_glmnet`; rerun
`_generate_dml.R` only when the DGP itself changes.

## References

- **Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E.,
  Hansen, C., Newey, W. & Robins, J. (2018).** Double/debiased
  machine learning for treatment and structural parameters. *The
  Econometrics Journal*, 21(1), C1–C68. [@chernozhukov2018double]
- **Bach, P., Chernozhukov, V., Kurz, M.S. & Spindler, M. (2022).**
  DoubleML — An Object-Oriented Implementation of Double Machine
  Learning in Python. *Journal of Machine Learning Research*,
  23(53), 1–6. [@bach2022doubleml]
- **Bach, P., Kurz, M.S., Chernozhukov, V., Spindler, M. & Klaassen,
  S. (2024).** DoubleML — An Object-Oriented Implementation of
  Double Machine Learning in R. *Journal of Statistical Software*,
  108(3), 1–56. DOI: [`10.18637/jss.v108.i03`](https://doi.org/10.18637/jss.v108.i03). [@bach2024doubleml]
