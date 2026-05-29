"""Validate the canonical core/_vcov sandwich wrappers against statsmodels.

This pins the NEW primitive (currently unwired) so that the future
one-estimator-at-a-time migration off the ~18 hand-rolled sandwich copies has
a trusted reference to migrate onto.
"""

import numpy as np
import pytest

from statspai.core._vcov import (
    cluster_robust_vcov,
    hc_vcov,
    cluster_correction_factor,
)


@pytest.fixture
def ols_data():
    rng = np.random.default_rng(0)
    n = 400
    X = np.column_stack([np.ones(n), rng.normal(size=n), rng.normal(size=n)])
    beta = np.array([1.0, 2.0, -0.5])
    y = X @ beta + rng.normal(size=n) * (1.0 + 0.5 * np.abs(X[:, 1]))  # heterosk.
    clusters = rng.integers(0, 25, size=n)
    bhat = np.linalg.solve(X.T @ X, X.T @ y)
    resid = y - X @ bhat
    return X, y, resid, clusters


def test_hc1_matches_statsmodels(ols_data):
    sm = pytest.importorskip("statsmodels.api")
    X, y, resid, _ = ols_data
    res = sm.OLS(y, X).fit()
    expected = res.cov_HC1
    got = hc_vcov(X, resid, hc_type="hc1")
    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-12)


def test_hc0_matches_statsmodels(ols_data):
    sm = pytest.importorskip("statsmodels.api")
    X, y, resid, _ = ols_data
    res = sm.OLS(y, X).fit()
    np.testing.assert_allclose(hc_vcov(X, resid, hc_type="hc0"),
                               res.cov_HC0, rtol=1e-10, atol=1e-12)


def test_cluster_stata_matches_statsmodels(ols_data):
    sm = pytest.importorskip("statsmodels.api")
    X, y, resid, clusters = ols_data
    res = sm.OLS(y, X).fit()
    rob = res.get_robustcov_results(cov_type="cluster", groups=clusters,
                                    use_correction=True)
    expected = rob.cov_params()
    got = cluster_robust_vcov(X, resid, clusters, correction="stata")
    np.testing.assert_allclose(got, expected, rtol=1e-8, atol=1e-10)


def test_correction_factor_relationships():
    # cgm = none * G/(G-1);  stata = cgm * (N-1)/(N-K)
    G, N, K = 25, 400, 3
    assert cluster_correction_factor(G, N, K, "none") == 1.0
    assert cluster_correction_factor(G, N, K, "cgm") == pytest.approx(G / (G - 1))
    assert cluster_correction_factor(G, N, K, "stata") == pytest.approx(
        (G / (G - 1)) * ((N - 1) / (N - K)))
    assert cluster_correction_factor(G, N, K, "stacked") == pytest.approx(
        (G / (G - 1)) * (N / (N - K)))


def test_dof_adjust_override(ols_data):
    X, y, resid, clusters = ols_data
    base = cluster_robust_vcov(X, resid, clusters, correction="none")
    scaled = cluster_robust_vcov(X, resid, clusters, dof_adjust=2.0)
    np.testing.assert_allclose(scaled, 2.0 * base, rtol=1e-12)


def test_single_cluster_does_not_explode(ols_data):
    X, y, resid, _ = ols_data
    one = np.zeros(len(resid), dtype=int)
    V = cluster_robust_vcov(X, resid, one, correction="stata")
    assert np.all(np.isfinite(V))


def test_unknown_correction_raises():
    with pytest.raises(ValueError, match="Unknown cluster correction"):
        cluster_correction_factor(10, 100, 3, "bogus")
