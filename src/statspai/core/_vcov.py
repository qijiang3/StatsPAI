"""Canonical cluster-robust / heteroskedasticity-robust VCOV wrappers.

CLAUDE.md §4 mandates one implementation of correctness-sensitive basics.
``core/_numba_kernels`` already owns the *meat* matrices (``cluster_meat``,
``sandwich_hc``); what is duplicated across ~18 estimators is the thin
*bread + finite-sample correction* layer:

    V = c * (X'X)^{-1} @ meat @ (X'X)^{-1}

reimplemented each time with a slightly different correction factor ``c``.
This module is that single wrapper, with the finite-sample correction made an
**explicit, named, documented parameter** rather than a scattered magic factor.

Status: NEW canonical primitive. It is intentionally NOT yet wired into the
call sites — those carry deliberately different corrections (Stata
``vce(cluster)`` vs Liang-Zeger vs CGM-asymptotic), so migration is a
parity-sensitive, one-estimator-at-a-time task (see
docs/rfc/vcov_consolidation.md). Each migrated site keeps its EXACT current
correction by passing the matching ``correction=``/``dof_adjust=``.

Correction factors (G = #clusters, N = #obs, K = #params):
  'none'         c = 1
  'cgm'          c = G / (G - 1)                         (Cameron-Gelbach-Miller)
  'stata'        c = (G/(G-1)) * ((N-1)/(N-K))           (Stata vce(cluster);
  'liang_zeger'  alias of 'stata'                         statsmodels default)
  'stacked'      c = (G/(G-1)) * (N/(N-K))               (used by stacked DiD)
A float ``dof_adjust`` overrides the named factor for any non-standard site.
"""

from __future__ import annotations

import numpy as np

from ._numba_kernels import cluster_meat, sandwich_hc

__all__ = ["cluster_robust_vcov", "hc_vcov", "cluster_correction_factor"]


def cluster_correction_factor(n_clusters: int, n_obs: int, n_params: int,
                              correction: str = "stata") -> float:
    """Finite-sample correction factor ``c`` for a cluster-robust sandwich."""
    G, N, K = int(n_clusters), int(n_obs), int(n_params)
    corr = correction.lower()
    if corr == "none":
        return 1.0
    if G <= 1:
        # G/(G-1) is undefined / explosive with one cluster; degrade to 1
        # and let the caller decide (a 1-cluster VCOV is not identified).
        return 1.0
    g_factor = G / (G - 1.0)
    if corr == "cgm":
        return g_factor
    if corr in ("stata", "liang_zeger", "liang-zeger"):
        denom = (N - K)
        return g_factor * ((N - 1.0) / denom) if denom > 0 else g_factor
    if corr == "stacked":
        denom = (N - K)
        return g_factor * (N / denom) if denom > 0 else g_factor
    raise ValueError(
        f"Unknown cluster correction {correction!r}; expected one of "
        "'none', 'cgm', 'stata', 'liang_zeger', 'stacked'."
    )


def cluster_robust_vcov(
    X: np.ndarray,
    residuals: np.ndarray,
    clusters: np.ndarray,
    *,
    correction: str = "stata",
    dof_adjust: float | None = None,
    XtX_inv: np.ndarray | None = None,
) -> np.ndarray:
    """One-way cluster-robust (sandwich) covariance matrix.

    ``V = c * (X'X)^{-1} @ meat @ (X'X)^{-1}`` where ``meat`` is the canonical
    ``cluster_meat`` and ``c`` is the finite-sample correction.

    Parameters
    ----------
    X : (n, k) design matrix (intercept column included by the caller).
    residuals : (n,) regression residuals.
    clusters : (n,) cluster labels (any hashable / integer dtype).
    correction : named finite-sample factor (see module docstring).
    dof_adjust : explicit float factor; overrides ``correction`` when given
        (use to reproduce a site's exact non-standard factor during migration).
    XtX_inv : optional precomputed (X'X)^{-1} (reuses the caller's bread).

    Returns
    -------
    (k, k) covariance matrix.
    """
    X = np.asarray(X, dtype=np.float64)
    residuals = np.asarray(residuals, dtype=np.float64)
    n, k = X.shape
    if XtX_inv is None:
        XtX_inv = np.linalg.inv(X.T @ X)
    meat = cluster_meat(X, residuals, np.asarray(clusters))
    n_clusters = int(np.unique(clusters).shape[0])
    if dof_adjust is not None:
        c = float(dof_adjust)
    else:
        c = cluster_correction_factor(n_clusters, n, k, correction)
    return c * (XtX_inv @ meat @ XtX_inv)


def hc_vcov(
    X: np.ndarray,
    residuals: np.ndarray,
    *,
    hc_type: str = "hc1",
    XtX_inv: np.ndarray | None = None,
) -> np.ndarray:
    """Heteroskedasticity-robust (HC0–HC3) covariance — thin wrapper over the
    canonical ``sandwich_hc`` kernel so call sites stop reimplementing it."""
    X = np.asarray(X, dtype=np.float64)
    residuals = np.asarray(residuals, dtype=np.float64)
    if XtX_inv is None:
        XtX_inv = np.linalg.inv(X.T @ X)
    return sandwich_hc(X, residuals, XtX_inv, hc_type=hc_type)
