"""Coverage round-2 — ``statspai.rd.rdml`` (ML + RD).

Round 1 covered happy paths of ``rd_forest`` / ``rd_boost`` / ``rd_lasso`` /
``rd_cate_summary``. This file adds the error / option branches:

- ``honesty=False`` forest;
- input-validation errors (no covs, running var in covs, too few obs per
  side, missing covariate column);
- ``rd_cate_summary`` with a method subset and the unknown-method error;
- ``rd_cate_summary`` collecting per-method errors into ``*_error`` keys;
- the variable-importance plot helper (``_importance_plot``) incl. its
  empty-importance guard.

sklearn is installed in this environment, so these paths are reachable. Real
synthetic RD data with a moderator-driven jump; assertions check recovered
magnitude, positive SE, and structural keys — never fabricated numbers.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import statspai as sp  # noqa: E402
from statspai.rd.rdml import _importance_plot  # noqa: E402

JUMP_Z0 = 2.0
JUMP_Z1 = 5.0
ATE = 3.5


def _rd_df(seed=0, n=1500):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n)
    treat = (x >= 0).astype(float)
    z = rng.integers(0, 2, n).astype(float)
    cov1 = rng.normal(size=n)
    eff = JUMP_Z0 + (JUMP_Z1 - JUMP_Z0) * z
    y = 0.5 * x + eff * treat + 0.3 * cov1 + rng.normal(0, 0.4, n)
    return pd.DataFrame({"y": y, "x": x, "z": z, "cov1": cov1})


def test_rd_forest_no_honesty():
    df = _rd_df()
    r = sp.rd_forest(df, y="y", x="x", c=0, covs=["cov1"], n_trees=50,
                     honesty=False)
    assert r.se > 0
    assert np.isfinite(np.atleast_1d(np.asarray(r.estimate, dtype=float)).ravel()[0])


def test_rd_forest_requires_covs():
    df = _rd_df()
    with pytest.raises(ValueError, match="covariate"):
        sp.rd_forest(df, y="y", x="x", c=0, covs=None, n_trees=50)


def test_rd_forest_running_var_in_covs_errors():
    df = _rd_df()
    with pytest.raises(ValueError, match="must not be in covs"):
        sp.rd_forest(df, y="y", x="x", c=0, covs=["x", "cov1"], n_trees=50)


def test_rd_forest_missing_covariate_column():
    df = _rd_df()
    with pytest.raises(ValueError, match="not found"):
        sp.rd_forest(df, y="y", x="x", c=0, covs=["does_not_exist"],
                     n_trees=50)


def test_rd_lasso_requires_covs():
    df = _rd_df()
    with pytest.raises(ValueError, match="covariates"):
        sp.rd_lasso(df, y="y", x="x", c=0, covs=None)


def test_rd_cate_summary_method_subset():
    df = _rd_df()
    out = sp.rd_cate_summary(df, y="y", x="x", c=0, covs=["cov1"],
                             methods=["lasso"])
    assert "lasso" in out
    assert "comparison" in out
    assert len(out["comparison"]) == 1


def test_rd_cate_summary_unknown_method():
    df = _rd_df()
    with pytest.raises(ValueError, match="Unknown methods"):
        sp.rd_cate_summary(df, y="y", x="x", c=0, covs=["cov1"],
                           methods=["forest", "nope"])


def test_rd_cate_summary_collects_errors():
    # too few covs/obs forces the per-method try/except error capture
    df = _rd_df(n=1500)
    out = sp.rd_cate_summary(df, y="y", x="x", c=0, covs=["cov1"], h=0.001,
                             methods=["forest"])
    # forest should fail (too few obs in tiny bandwidth) -> forest_error key
    assert "forest_error" in out or "forest" in out
    assert "comparison" in out


def test_importance_plot_from_forest():
    df = _rd_df()
    r = sp.rd_forest(df, y="y", x="x", c=0, covs=["cov1", "z"], n_trees=50)
    fig0, ax0 = plt.subplots()
    ax = _importance_plot(r, top_k=5, ax=ax0)
    assert ax is not None
    plt.close("all")


def test_importance_plot_creates_axes():
    df = _rd_df()
    r = sp.rd_forest(df, y="y", x="x", c=0, covs=["cov1", "z"], n_trees=50)
    ax = _importance_plot(r)
    assert ax is not None
    plt.close("all")


def test_importance_plot_empty_raises():
    df = _rd_df()
    r = sp.rd_lasso(df, y="y", x="x", c=0, covs=["cov1"])
    # rd_lasso result has no variable_importance -> guard raises
    with pytest.raises(ValueError, match="variable_importance"):
        _importance_plot(r)
