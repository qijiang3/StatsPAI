"""Behavioral regression tests for silent-degradation correctness fixes.

CLAUDE.md §7 forbids silently swallowing a numerical failure and returning a
degraded estimate as if it were the requested one. These tests lock in the
fixes that turn previously-silent degradations into loud warnings + an audit
trail in the result's ``model_info`` / ``diagnostics``.

Each test forces the failure path deterministically (monkeypatching the
fragile sub-fit to fail) so it does not depend on a specific BLAS / statsmodels
separation behavior.
"""

import importlib
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp


# ---------------------------------------------------------------------------
# WS-A1: covariate-adjusted logit silently reverting to the unadjusted
#        marginal mean (front_door, principal_strat).
# ---------------------------------------------------------------------------

def _front_door_data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=n)
    D = (rng.uniform(size=n) < 0.5).astype(int)
    M = (rng.uniform(size=n) < 0.3 + 0.3 * D).astype(int)
    Y = 1.0 * M + 0.5 * X + rng.normal(size=n)
    return pd.DataFrame({"Y": Y, "D": D, "M": M, "X": X})


def test_front_door_warns_when_mediator_logit_degrades(monkeypatch):
    """When the covariate-adjusted mediator logit fails, front_door must warn
    that the reported ATE is no longer covariate-adjusted (not silently swap
    in the marginal P(M=1))."""
    # NB: ``statspai.inference.front_door`` resolves to the re-exported
    # function, so fetch the real module via sys.modules.
    fd = importlib.import_module("statspai.inference.front_door")

    df = _front_door_data()

    # Force the mediator logit to fail on every fit -> marginal-mean fallback.
    monkeypatch.setattr(fd, "_logit_fit", lambda y, X: None)

    with pytest.warns(RuntimeWarning, match="no longer covariate-adjusted"):
        res = sp.front_door(
            df, y="Y", treat="D", mediator="M", covariates=["X"],
            n_boot=10, seed=1,
        )
    assert res.model_info["mediator_model_degraded"] is True
    assert res.model_info["mediator_model_fallback_arms"] == 2


def test_front_door_clean_run_records_no_degradation():
    """A healthy fit must report mediator_model_degraded=False."""
    df = _front_door_data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.front_door(
            df, y="Y", treat="D", mediator="M", covariates=["X"],
            n_boot=10, seed=1,
        )
    assert res.model_info["mediator_model_degraded"] is False
    assert res.model_info["mediator_model_fallback_arms"] == 0


def _principal_data(n=600, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=n)
    D = (rng.uniform(size=n) < 0.5).astype(int)
    # S = post-treatment intermediate (e.g. survival / employment)
    S = (rng.uniform(size=n) < 0.5 + 0.2 * D + 0.1 * X).astype(int)
    Y = 2.0 * S + 0.5 * X + rng.normal(size=n)
    return pd.DataFrame({"Y": Y, "D": D, "S": S, "X": X})


def test_principal_score_warns_when_logit_degrades(monkeypatch):
    """principal_strat(method='principal_score') must warn when the
    covariate-adjusted principal-score logit reverts to the marginal."""
    ps = importlib.import_module("statspai.principal_strat.principal_strat")

    df = _principal_data()
    monkeypatch.setattr(ps, "_logit_safe", lambda y, X: None)

    with pytest.warns(RuntimeWarning, match="no longer covariate-adjusted"):
        res = sp.principal_strat(
            df, y="Y", treat="D", strata="S", covariates=["X"],
            method="principal_score", n_boot=10, seed=1,
        )
    assert res.model_info["principal_score_degraded"] is True
