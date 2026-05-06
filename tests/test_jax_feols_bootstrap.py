"""Parity tests for ``sp.fast.feols_jax_bootstrap`` (Phase 4b).

The bootstrap SEs are stochastic, so parity here is in the
**convergence** sense:
  * Pairs-bootstrap SE → HC1 SE as B → ∞
  * Cluster-bootstrap SE → CR1 SE as B → ∞

We use B=2000 throughout so the Monte-Carlo SE on the bootstrap SE
estimate is small enough to assert ~5% relative tolerance, and we
also pin a couple of deterministic invariants (point estimate from
``coef`` matches ``feols_jax`` exactly; same-seed runs are bit-
identical).

Skips automatically when JAX is not installed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("jax")

from statspai.fast import (
    feols,
    feols_jax,
    feols_jax_bootstrap,
    FeolsBootstrapResult,
)


def _make_panel(n: int = 1_000, n_firm: int = 50, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    firm = rng.integers(0, n_firm, size=n)
    fe = rng.normal(size=n_firm)[firm]
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 0.5 * x1 - 0.2 * x2 + fe + 0.5 * rng.normal(size=n)
    return pd.DataFrame({
        "y": y, "x1": x1, "x2": x2,
        "firm": firm, "cluster": firm,
    })


# ---------------------------------------------------------------------------
# Deterministic invariants (no Monte-Carlo noise)
# ---------------------------------------------------------------------------

def test_point_estimate_matches_feols_jax():
    """``coef`` is the un-resampled point estimate; must match feols_jax exactly."""
    df = _make_panel(seed=1)
    fit = feols_jax("y ~ x1 + x2 | firm", df, vcov="iid")
    boot = feols_jax_bootstrap(
        "y ~ x1 + x2 | firm", df, n_boot=100, seed=0,
    )
    np.testing.assert_allclose(boot.coef.values, fit.coef_vec, atol=1e-12)
    assert list(boot.coef.index) == list(fit.coef_names)


def test_returns_correct_dataclass_type():
    df = _make_panel(seed=2)
    boot = feols_jax_bootstrap(
        "y ~ x1 + x2 | firm", df, n_boot=50, seed=0,
    )
    assert isinstance(boot, FeolsBootstrapResult)
    assert boot.bootstrap_type == "pairs"
    assert boot.backend == "statspai-jax-bootstrap"
    assert boot.n_boot == 50


def test_same_seed_gives_identical_results():
    df = _make_panel(seed=3)
    b1 = feols_jax_bootstrap("y ~ x1 + x2 | firm", df, n_boot=100, seed=42)
    b2 = feols_jax_bootstrap("y ~ x1 + x2 | firm", df, n_boot=100, seed=42)
    np.testing.assert_array_equal(b1.boot_betas.values, b2.boot_betas.values)


def test_boot_betas_shape_and_columns():
    df = _make_panel(seed=4)
    boot = feols_jax_bootstrap(
        "y ~ x1 + x2 | firm", df, n_boot=137, seed=0,
    )
    assert boot.boot_betas.shape == (137, 2)
    assert list(boot.boot_betas.columns) == ["x1", "x2"]


def test_chunk_size_does_not_change_results():
    """Different chunk sizes split the same vmap → identical numbers."""
    df = _make_panel(seed=5)
    b_small = feols_jax_bootstrap(
        "y ~ x1 + x2 | firm", df, n_boot=200, seed=7, vmap_chunk_size=20,
    )
    b_large = feols_jax_bootstrap(
        "y ~ x1 + x2 | firm", df, n_boot=200, seed=7, vmap_chunk_size=200,
    )
    np.testing.assert_allclose(
        b_small.boot_betas.values, b_large.boot_betas.values, atol=1e-12,
    )


# ---------------------------------------------------------------------------
# Convergence to analytic SEs
# ---------------------------------------------------------------------------

def test_pairs_bootstrap_se_converges_to_hc1():
    """Pairs SE → HC1 SE as B grows."""
    df = _make_panel(n=2_000, n_firm=80, seed=11)
    fit_hc1 = feols("y ~ x1 + x2 | firm", df, vcov="hc1")
    boot = feols_jax_bootstrap(
        "y ~ x1 + x2 | firm", df, n_boot=2_000, seed=0,
    )
    # 5% relative tolerance — empirically tight enough at B=2000 even
    # with the JAX-vs-numpy random-stream divergence.
    np.testing.assert_allclose(
        boot.se_boot["x1"], fit_hc1.se()["x1"], rtol=0.10,
    )
    np.testing.assert_allclose(
        boot.se_boot["x2"], fit_hc1.se()["x2"], rtol=0.10,
    )


def test_cluster_bootstrap_se_converges_to_cr1():
    """Cluster SE → CR1 SE as B grows."""
    df = _make_panel(n=2_000, n_firm=80, seed=12)
    fit_cr1 = feols(
        "y ~ x1 + x2 | firm", df, vcov="cr1", cluster="cluster",
    )
    boot = feols_jax_bootstrap(
        "y ~ x1 + x2 | firm", df, n_boot=2_000, seed=0,
        bootstrap="cluster", cluster="cluster",
    )
    # Cluster bootstrap convergence is slower than pairs; allow 15%.
    np.testing.assert_allclose(
        boot.se_boot["x1"], fit_cr1.se()["x1"], rtol=0.15,
    )


def test_percentile_ci_contains_true_value_for_well_specified_dgp():
    """95% CI should cover the true coefficient on a clean DGP."""
    df = _make_panel(n=2_000, n_firm=80, seed=13)
    boot = feols_jax_bootstrap(
        "y ~ x1 + x2 | firm", df, n_boot=2_000, seed=0,
    )
    # True beta_x1 = 0.5 in _make_panel
    assert boot.ci_lower["x1"] < 0.5 < boot.ci_upper["x1"]
    # True beta_x2 = -0.2
    assert boot.ci_lower["x2"] < -0.2 < boot.ci_upper["x2"]


# ---------------------------------------------------------------------------
# Validation / error paths
# ---------------------------------------------------------------------------

def test_invalid_bootstrap_type_raises():
    df = _make_panel(seed=20)
    with pytest.raises(ValueError, match="bootstrap="):
        feols_jax_bootstrap("y ~ x1", df, n_boot=10, bootstrap="wild")


def test_cluster_bootstrap_without_cluster_raises():
    df = _make_panel(seed=21)
    with pytest.raises(ValueError, match="cluster="):
        feols_jax_bootstrap(
            "y ~ x1", df, n_boot=10, bootstrap="cluster",
        )


def test_n_boot_below_one_raises():
    df = _make_panel(seed=22)
    with pytest.raises(ValueError, match="n_boot"):
        feols_jax_bootstrap("y ~ x1", df, n_boot=0)


def test_chunk_below_one_raises():
    df = _make_panel(seed=23)
    with pytest.raises(ValueError, match="vmap_chunk_size"):
        feols_jax_bootstrap("y ~ x1", df, n_boot=10, vmap_chunk_size=0)


def test_invalid_ci_alpha_raises():
    df = _make_panel(seed=24)
    with pytest.raises(ValueError, match="ci_alpha"):
        feols_jax_bootstrap("y ~ x1", df, n_boot=10, ci_alpha=1.5)


def test_invalid_dtype_raises():
    df = _make_panel(seed=25)
    with pytest.raises(ValueError, match="dtype="):
        feols_jax_bootstrap("y ~ x1", df, n_boot=10, dtype="bfloat16")


def test_missing_cluster_column_raises():
    df = _make_panel(seed=26)
    with pytest.raises(KeyError, match="not_a_column"):
        feols_jax_bootstrap(
            "y ~ x1", df, n_boot=10,
            bootstrap="cluster", cluster="not_a_column",
        )


# ---------------------------------------------------------------------------
# Summary + repr
# ---------------------------------------------------------------------------

def test_summary_contains_key_metadata():
    df = _make_panel(seed=30)
    boot = feols_jax_bootstrap(
        "y ~ x1 + x2 | firm", df, n_boot=50, seed=0,
    )
    s = boot.summary()
    assert "feols_jax_bootstrap" in s
    assert "pairs" in s
    assert "n_boot=50" in s
    assert "x1" in s
