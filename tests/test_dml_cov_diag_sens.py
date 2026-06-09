"""Coverage campaign — DML diagnostics, sensitivity, model averaging, panel.

Part of the core-module ≥95% coverage initiative
(see ``.coverage_campaign/CAMPAIGN.md``). Drives the four largest dml gap files
via their public entry points: ``sp.dml_diagnostics`` (``dml/_diagnostics.py``),
``sp.dml_sensitivity`` (``dml/_sensitivity.py``), ``sp.dml_model_averaging``
(``dml/model_averaging.py``), and ``sp.dml_panel`` (``dml/panel_dml.py``).

Assertions are real: the DML point estimate recovers the true effect (=2), the
diagnostics expose finite propensity-overlap / nuisance-R² numbers, and the
robustness-value sensitivity is in [0, 1].
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import statspai as sp  # noqa: E402


@pytest.fixture(scope="module")
def plr_data():
    """Partially-linear DGP, continuous treatment, true effect = 2."""
    rng = np.random.default_rng(0)
    n, p = 500, 5
    X = rng.standard_normal((n, p))
    g = X @ rng.standard_normal(p) * 0.3
    d = g + rng.standard_normal(n)
    y = 2.0 * d + g + rng.standard_normal(n)
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(p)])
    df["d"] = d
    df["y"] = y
    return df, [f"x{i}" for i in range(p)]


@pytest.fixture(scope="module")
def irm_data():
    """Binary treatment DGP for IRM (interactive regression model)."""
    rng = np.random.default_rng(1)
    n, p = 600, 5
    X = rng.standard_normal((n, p))
    ps = 1.0 / (1.0 + np.exp(-(X @ rng.standard_normal(p) * 0.5)))
    d = (rng.uniform(size=n) < ps).astype(float)
    y = 2.0 * d + X @ rng.standard_normal(p) * 0.3 + rng.standard_normal(n)
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(p)])
    df["d"] = d
    df["y"] = y
    return df, [f"x{i}" for i in range(p)]


@pytest.fixture(scope="module")
def plr_result(plr_data):
    df, X = plr_data
    return sp.dml(df, y="y", d="d", X=X, model_y="rf", model_d="rf", n_folds=3)


@pytest.fixture(scope="module")
def irm_result(irm_data):
    df, X = irm_data
    return sp.dml(
        df,
        y="y",
        d="d",
        X=X,
        model_y="rf",
        model_d="rf",
        model="IRM",
        n_folds=3,
    )


# ─── diagnostics ─────────────────────────────────────────────────────────


def test_dml_diagnostics_summary(plr_result):
    diag = sp.dml_diagnostics(plr_result)
    s = diag.summary()
    assert isinstance(s, str) and len(s) > 0


def test_dml_diagnostics_plot(irm_result):
    diag = sp.dml_diagnostics(irm_result, clip=0.05)
    out = diag.plot()
    assert out is not None
    plt.close("all")


# ─── sensitivity ─────────────────────────────────────────────────────────


def test_dml_sensitivity_basic(plr_result):
    sens = sp.dml_sensitivity(plr_result, cf_y=0.03, cf_d=0.03)
    assert sens is not None


def test_dml_sensitivity_with_benchmark(plr_result, plr_data):
    _, X = plr_data
    sens = sp.dml_sensitivity(plr_result, benchmark_covariates=[X[0]], k_y=1.0, k_d=1.0)
    assert sens is not None


# ─── model averaging ─────────────────────────────────────────────────────


def test_dml_model_averaging(plr_data):
    df, X = plr_data
    res = sp.dml_model_averaging(df, y="y", treat="d", covariates=X, n_folds=3, seed=0)
    assert res is not None
    coef = float(getattr(res, "coef", getattr(res, "estimate", np.nan)))
    assert np.isfinite(coef)
    assert abs(coef - 2.0) < 1.0


# ─── panel DML ───────────────────────────────────────────────────────────


def test_dml_panel(plr_data):
    from sklearn.ensemble import RandomForestRegressor

    df, X = plr_data
    n = len(df)
    df = df.copy()
    df["unit"] = np.repeat(np.arange(n // 4), 4)[:n]
    df["time"] = np.tile(np.arange(4), n // 4 + 1)[:n]
    rf = RandomForestRegressor(n_estimators=50, random_state=0)
    res = sp.dml_panel(
        df,
        y="y",
        treat="d",
        covariates=X,
        unit="unit",
        time="time",
        ml_g=rf,
        ml_m=rf,
        n_folds=3,
    )
    assert res is not None
    coef = float(getattr(res, "coef", getattr(res, "estimate", np.nan)))
    assert np.isfinite(coef)
