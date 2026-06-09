"""Coverage for previously-untested partial-identification and diagnostic
helpers: ``sp.attrition_bounds`` (Lee bounds), ``sp.breakdown_frontier``,
``sp.moran_residuals`` and ``sp.propensity_score``.

Assertions are restricted to laws that hold regardless of the random draw —
bounds bracketing/widening, a spatial field producing positive Moran's I,
and propensity scores being monotone in the treatment-driving covariate —
so the tests are deterministic in spirit even where the data is simulated.
"""

import numpy as np
import pandas as pd
import pytest

import statspai as sp


# --------------------------------------------------------------------------
# attrition_bounds — Lee (2009) trimming bounds
# --------------------------------------------------------------------------
def test_attrition_bounds_bracket_and_report_attrition():
    rng = np.random.RandomState(2)
    n = 400
    df = pd.DataFrame({"y": rng.randn(n), "t": rng.binomial(1, 0.5, n)})
    df["obs"] = rng.binomial(1, 0.85, n)
    df.loc[df["obs"] == 0, "y"] = np.nan

    res = sp.attrition_bounds(df, "y", "t", observed="obs")
    assert res["lower_bound"] <= res["upper_bound"]
    # Reported attrition rate equals the share of unobserved outcomes.
    assert res["attrition_rate"] == pytest.approx(1 - df["obs"].mean(), abs=1e-9)
    assert res["n_total"] == n


def test_no_attrition_collapses_the_bounds():
    rng = np.random.RandomState(3)
    n = 300
    df = pd.DataFrame(
        {
            "y": rng.randn(n),
            "t": rng.binomial(1, 0.5, n),
            "obs": np.ones(n, dtype=int),
        }
    )
    res = sp.attrition_bounds(df, "y", "t", observed="obs")
    # With nobody missing, Lee bounds are point-identified: lower == upper.
    assert res["lower_bound"] == pytest.approx(res["upper_bound"], abs=1e-9)
    assert res["attrition_rate"] == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------
# breakdown_frontier — sensitivity bounds
# --------------------------------------------------------------------------
def test_breakdown_frontier_widens_with_violation():
    narrow = sp.breakdown_frontier(estimate=0.5, se=0.1, max_violation=0.1)
    wide = sp.breakdown_frontier(estimate=0.5, se=0.1, max_violation=0.5)
    assert narrow.lower <= narrow.upper
    assert wide.lower <= wide.upper
    assert narrow.width < wide.width


def test_breakdown_frontier_brackets_the_estimate():
    b = sp.breakdown_frontier(estimate=0.5, se=0.1, max_violation=0.2)
    assert b.lower <= 0.5 <= b.upper


# --------------------------------------------------------------------------
# moran_residuals — spatial autocorrelation of residuals
# --------------------------------------------------------------------------
def _ring_W(n):
    W = np.zeros((n, n))
    for i in range(n - 1):
        W[i, i + 1] = W[i + 1, i] = 1.0
    W[0, n - 1] = W[n - 1, 0] = 1.0
    return W


def test_moran_detects_spatial_structure():
    n = 40
    # A smooth field over the ring is strongly positively autocorrelated.
    smooth = np.sin(2 * np.pi * np.arange(n) / n)
    I, p = sp.moran_residuals(smooth, _ring_W(n))
    assert I > 0.0
    assert 0.0 <= p <= 1.0
    assert p < 0.05  # smooth field -> significant positive autocorrelation


def test_moran_returns_finite_statistic_for_noise():
    rng = np.random.RandomState(7)
    n = 40
    I, p = sp.moran_residuals(rng.randn(n), _ring_W(n))
    assert np.isfinite(I)
    assert 0.0 <= p <= 1.0


# --------------------------------------------------------------------------
# propensity_score
# --------------------------------------------------------------------------
def test_propensity_scores_are_probabilities_and_monotone():
    rng = np.random.RandomState(5)
    n = 500
    df = pd.DataFrame({"x": rng.randn(n)})
    df["t"] = (df["x"] + 0.5 * rng.randn(n) > 0).astype(int)
    ps = sp.propensity_score(df, "t", ["x"])
    assert len(ps) == n
    assert float(ps.min()) > 0.0 and float(ps.max()) < 1.0
    # Treatment is driven by x, so treated units carry higher average scores.
    assert ps[df["t"] == 1].mean() > ps[df["t"] == 0].mean()
