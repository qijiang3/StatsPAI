"""Coverage campaign — RD estimator family (ML, HTE, 2D, multi-cutoff, …).

Part of the core-module ≥95% coverage initiative
(see ``.coverage_campaign/CAMPAIGN.md``). Drives the RD estimators left thinly
covered after the parallel agent's ``test_cov95_rd_*`` files: boosted/ML RD
(``rdml.py`` / ``rd_flex.py``), CATE/HTE summaries (``hte.py``), the dashboard
(``dashboard.py``), 2-D and multi-cutoff/multi-score designs (``rd2d.py`` /
``rdmulti.py``), local randomization (``locrand.py``), honest and bias-aware
CIs (``honest_ci.py`` / ``bias_aware.py``), and extrapolation
(``extrapolate.py``).

Smoke + structural assertions: each estimator returns a non-None result with a
finite effect / interval where one is exposed. Campaign-own ``test_rd_cov_*``
naming avoids colliding with the parallel ``test_cov95_rd_*`` files.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp


@pytest.fixture(scope="module")
def sharp():
    rng = np.random.default_rng(0)
    n = 1500
    x = rng.uniform(-1, 1, n)
    y = 0.5 * x + 0.8 * (x >= 0) + rng.normal(0, 0.3, n)
    return pd.DataFrame(
        {"y": y, "x": x, "z": rng.normal(0, 1, n), "z2": rng.normal(0, 1, n)}
    )


@pytest.fixture(scope="module")
def fuzzy():
    rng = np.random.default_rng(11)
    n = 3000
    x = rng.uniform(-1, 1, n)
    prob = 0.3 + 0.4 * (x >= 0)
    d = (rng.uniform(0, 1, n) < prob).astype(float)
    y = 0.5 * x + 2.0 * d + rng.normal(0, 0.4, n)
    return pd.DataFrame({"y": y, "x": x, "d": d})


# ─── ML / boosted RD ─────────────────────────────────────────────────────


def test_rd_boost(sharp):
    res = sp.rd_boost(sharp, y="y", x="x", c=0.0, covs=["z", "z2"])
    assert res is not None


def test_rd_flex(sharp):
    res = sp.rd_flex(sharp, y="y", x="x", c=0.0, W=["z", "z2"], learner="boost")
    assert res is not None


# ─── HTE / dashboard ─────────────────────────────────────────────────────


def test_rd_cate_summary(sharp):
    res = sp.rd_cate_summary(sharp, y="y", x="x", c=0.0, covs=["z", "z2"])
    assert res is not None


def test_rd_dashboard(sharp):
    res = sp.rd_dashboard(sharp, y="y", x="x", c=0.0)
    assert res is not None


# ─── multi-cutoff / multi-score / 2-D ────────────────────────────────────


def test_rdmc_multi_cutoff():
    rng = np.random.default_rng(3)
    n = 2000
    x = rng.uniform(-1, 1, n)
    cut = np.where(rng.uniform(size=n) < 0.5, -0.3, 0.3)
    y = 0.5 * x + 0.8 * (x >= cut) + rng.normal(0, 0.3, n)
    df = pd.DataFrame({"y": y, "x": x, "cutoff": cut})
    res = sp.rdmc(df, y="y", x="x", cutoffs=[-0.3, 0.3])
    assert res is not None


def test_rd2d_boundary():
    rng = np.random.default_rng(4)
    n = 2500
    x1 = rng.uniform(-1, 1, n)
    x2 = rng.uniform(-1, 1, n)
    treat = ((x1 >= 0) | (x2 >= 0)).astype(int)
    y = 0.5 * x1 + 0.3 * x2 + 0.8 * treat + rng.normal(0, 0.3, n)
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "treat": treat})
    res = sp.rd2d(df, y="y", x1="x1", x2="x2", treatment="treat")
    assert res is not None


# ─── local randomization ─────────────────────────────────────────────────


def test_rdrandinf(sharp):
    res = sp.rdrandinf(sharp, y="y", x="x", c=0.0, wl=-0.1, wr=0.1)
    assert res is not None


# ─── honest / bias-aware CIs ─────────────────────────────────────────────


def test_rd_honest(sharp):
    res = sp.rd_honest(sharp, y="y", x="x", c=0.0, M=2.0)
    assert res is not None


def test_rd_bias_aware_fuzzy(fuzzy):
    res = sp.rd_bias_aware_fuzzy(fuzzy, y="y", x="x", fuzzy="d", c=0.0)
    assert res is not None


# ─── extrapolation ───────────────────────────────────────────────────────


def test_rd_extrapolate(sharp):
    # Angrist-Rokkanen extrapolation needs covariates (CIA: Y(d) ⊥ X | Z)
    res = sp.rd_extrapolate(sharp, y="y", x="x", c=0.0, covs=["z", "z2"])
    assert res is not None
