"""External parity: ``sp.dml`` vs ``DoubleML/doubleml-for-py``.

This module pins ``sp.dml`` against the upstream Python reference
implementation (DoubleML by Bach, Chernozhukov, Kurz & Spindler, 2022,
JMLR 23(53)) — complementary to the R-side check in
``tests/reference_parity/test_dml_parity.py``.

Both engines consume the in-repo seed=42 DGP fixture
(``tests/reference_parity/_fixtures/dml_data.csv``, n=1000, p=10,
true theta=0.5) and identical scikit-learn nuisance learners
(``LassoCV(cv=5)`` for regression, ``LogisticRegressionCV(cv=5)`` for
binary propensity), so any numerical divergence reflects a genuine
implementation difference rather than learner choice or DGP noise.

Skipped automatically when ``doubleml`` is not installed — this is an
optional pin (DoubleML is not a runtime dependency of StatsPAI).

Tolerance discipline
--------------------
- PLR (continuous d): coefficient agreement to 1e-3 absolute and SE
  agreement to 1e-3 absolute. The two implementations share the same
  Neyman-orthogonal score and the same scikit-learn folds under a
  fixed seed, so we expect bit-for-bit agreement at typical
  ``LassoCV`` precision.
- IRM (binary d, AIPW): coefficient agreement to 0.05 absolute. The
  two implementations make minor differences in propensity-score
  trimming and AIPW score normalization; on this DGP both estimates
  are statistically indistinguishable from zero (the truth).

References
----------
- Bach, P., Chernozhukov, V., Kurz, M.S., Spindler, M. (2022).
  DoubleML — An Object-Oriented Implementation of Double Machine
  Learning in Python. *Journal of Machine Learning Research*,
  23(53), 1-6. [@bach2022doubleml]
- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen,
  C., Newey, W. & Robins, J. (2018). Double/debiased machine learning
  for treatment and structural parameters. *The Econometrics
  Journal*, 21(1), C1-C68. [@chernozhukov2018double]
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest

import statspai as sp

doubleml = pytest.importorskip("doubleml")
from sklearn.linear_model import LassoCV, LogisticRegressionCV


_FIXTURE = (
    pathlib.Path(__file__).parent.parent
    / "reference_parity"
    / "_fixtures"
    / "dml_data.csv"
)


@pytest.fixture(scope="module")
def dml_data() -> pd.DataFrame:
    """Same csv the R reference parity test consumes."""
    return pd.read_csv(_FIXTURE)


@pytest.fixture(scope="module")
def x_cols() -> list[str]:
    return [f"x{i}" for i in range(1, 11)]


def test_plr_matches_doubleml_for_py(dml_data, x_cols):
    """sp.dml(model='plr') ≡ doubleml.DoubleMLPLR under identical learners."""
    np.random.seed(42)
    sp_res = sp.dml(
        data=dml_data, y="y", treat="d", covariates=x_cols,
        model="plr",
        ml_g=LassoCV(cv=5, random_state=42),
        ml_m=LassoCV(cv=5, random_state=42),
        n_folds=5, n_rep=1, random_state=42,
    )
    sp_coef = float(sp_res.estimate)
    sp_se = float(sp_res.se if np.isscalar(sp_res.se) else np.asarray(sp_res.se)[0])

    np.random.seed(42)
    dml_data_obj = doubleml.DoubleMLData(dml_data, y_col="y", d_cols="d", x_cols=x_cols)
    dml_plr = doubleml.DoubleMLPLR(
        dml_data_obj,
        ml_l=LassoCV(cv=5, random_state=42),
        ml_m=LassoCV(cv=5, random_state=42),
        n_folds=5, n_rep=1,
    )
    dml_plr.fit()
    ref_coef = float(dml_plr.coef[0])
    ref_se = float(dml_plr.se[0])

    assert abs(sp_coef - ref_coef) < 1e-3, (
        f"PLR coef mismatch: sp.dml={sp_coef:.6f}, doubleml-py={ref_coef:.6f}"
    )
    assert abs(sp_se - ref_se) < 1e-3, (
        f"PLR SE mismatch: sp.dml={sp_se:.6f}, doubleml-py={ref_se:.6f}"
    )


def test_irm_matches_doubleml_for_py(dml_data, x_cols):
    """sp.dml(model='irm') agrees with doubleml.DoubleMLIRM (binary D, AIPW)."""
    np.random.seed(42)
    sp_res = sp.dml(
        data=dml_data, y="y", treat="d_bin", covariates=x_cols,
        model="irm",
        ml_g=LassoCV(cv=5, random_state=42),
        ml_m=LogisticRegressionCV(cv=5, random_state=42, max_iter=2000),
        n_folds=5, n_rep=1, random_state=42,
    )
    sp_coef = float(sp_res.estimate)

    np.random.seed(42)
    dml_data_obj = doubleml.DoubleMLData(
        dml_data, y_col="y", d_cols="d_bin", x_cols=x_cols
    )
    dml_irm = doubleml.DoubleMLIRM(
        dml_data_obj,
        ml_g=LassoCV(cv=5, random_state=42),
        ml_m=LogisticRegressionCV(cv=5, random_state=42, max_iter=2000),
        n_folds=5, n_rep=1,
    )
    dml_irm.fit()
    ref_coef = float(dml_irm.coef[0])

    # IRM differs by AIPW score normalization / propensity trimming; both
    # implementations land statistically at zero on this DGP. Allow 0.05
    # absolute (= 2/3 of one SE on this fixture).
    assert abs(sp_coef - ref_coef) < 0.05, (
        f"IRM coef diverged beyond 0.05 absolute: "
        f"sp.dml={sp_coef:.6f}, doubleml-py={ref_coef:.6f}"
    )
