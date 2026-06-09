"""Tier D analytic special-case tests — proxy-variable production functions.

Part of the P1 "Tier D analytic special-cases" campaign (see
``.tierd_campaign/CAMPAIGN.md``). The short aliases ``levpet`` / ``opreg`` were
graded ``untested`` by ``scripts/tierd_classify.py``. On a data-generating
process that *satisfies* the proxy-variable timing assumptions — labour with an
independent exogenous source of variation (so it is identified separately from
productivity), and a proxy monotone in (productivity, capital) — the
Levinsohn-Petrin and Olley-Pakes estimators recover the known Cobb-Douglas
output elasticities. Each alias must also dispatch identically to its underlying
estimator.

NB: ``sp.blp`` (the third structural P1 estimator) is **deferred** — the Tier D
probe surfaced a real bug (``_gmm_objective`` called with ``maxiter=`` instead
of ``maxiter_inner=``); see ``.tierd_campaign/BUG_blp_gmm_objective_maxiter.md``.
Per the campaign red line it is reported, not silently fixed.

Entry points covered:
    sp.levpet  -> sp.levinsohn_petrin (Levinsohn-Petrin 2003)
    sp.opreg   -> sp.olley_pakes      (Olley-Pakes 1996)

Purely additive — no estimator numerics changed (campaign red line).
"""

import numpy as np
import pandas as pd
import pytest

import statspai as sp

BETA_L, BETA_K = 0.60, 0.35


def _identified_panel(seed=0, n_firms=300, n_periods=15, rho=0.7):
    """Cobb-Douglas panel satisfying the LP/OP assumptions.

    Crucially labour ``l`` is drawn *exogenously* (independent of the
    productivity innovation), so it is identified separately from omega — this
    avoids the Ackerberg-Caves-Frazer collinearity that biases the estimators
    when labour responds to contemporaneous productivity.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for fid in range(n_firms):
        omega = rng.normal(0.0, 0.2 / np.sqrt(1 - rho**2))
        k = rng.normal(0.0, 0.5)
        for t in range(n_periods):
            omega = rho * omega + rng.normal(0.0, 0.2)
            ell = rng.normal(0.5, 0.4)  # exogenous labour
            m = 0.8 * omega + 0.5 * k + rng.normal(0.0, 0.05)  # materials proxy
            i = np.exp(0.5 + 0.6 * omega + 0.3 * k + rng.normal(0.0, 0.05))
            y = BETA_L * ell + BETA_K * k + omega + rng.normal(0.0, 0.10)
            rows.append(
                {"id": fid, "year": t, "y": y, "l": ell, "k": k, "m": m, "i": i}
            )
            k = 0.9 * k + 0.1 * np.log(i + 1e-6)
    return pd.DataFrame(rows)


class TestLevinsohnPetrinAnalytic:

    def test_recovers_cobb_douglas_elasticities(self):
        df = _identified_panel()
        res = sp.levpet(
            df, output="y", free="l", state="k", proxy="m", panel_id="id", time="year"
        )
        assert res.coef["l"] == pytest.approx(BETA_L, abs=0.08)
        assert res.coef["k"] == pytest.approx(BETA_K, abs=0.10)

    def test_alias_equals_levinsohn_petrin(self):
        df = _identified_panel()
        a = sp.levpet(
            df, output="y", free="l", state="k", proxy="m", panel_id="id", time="year"
        )
        b = sp.levinsohn_petrin(
            df, output="y", free="l", state="k", proxy="m", panel_id="id", time="year"
        )
        assert a.coef["l"] == b.coef["l"] and a.coef["k"] == b.coef["k"]

    def test_first_stage_fits(self):
        df = _identified_panel()
        res = sp.levpet(
            df, output="y", free="l", state="k", proxy="m", panel_id="id", time="year"
        )
        assert res.diagnostics["stage1_r2"] > 0.5


class TestOlleyPakesAnalytic:

    def test_recovers_cobb_douglas_elasticities(self):
        df = _identified_panel()
        res = sp.opreg(
            df, output="y", free="l", state="k", proxy="i", panel_id="id", time="year"
        )
        assert res.coef["l"] == pytest.approx(BETA_L, abs=0.08)
        assert res.coef["k"] == pytest.approx(BETA_K, abs=0.10)

    def test_alias_equals_olley_pakes(self):
        df = _identified_panel()
        a = sp.opreg(
            df, output="y", free="l", state="k", proxy="i", panel_id="id", time="year"
        )
        b = sp.olley_pakes(
            df, output="y", free="l", state="k", proxy="i", panel_id="id", time="year"
        )
        assert a.coef["l"] == b.coef["l"] and a.coef["k"] == b.coef["k"]
