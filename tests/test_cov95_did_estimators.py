"""Coverage tests for several statspai.did estimators: did_2x2 options,
overlap_weighted_did, sun_abraham options, aggte aggregation types, lp_did,
cohort_anchored, gardner_did, did_multiplegt_dyn, ddd_heterogeneous,
did_misclassified, continuous_did, harvest_did."""
import numpy as np
import pandas as pd
import pytest

import statspai as sp


# ----------------------------------------------------------------------
# fixtures
# ----------------------------------------------------------------------

def _data_2x2(seed=0, n=1500):
    rng = np.random.default_rng(seed)
    treat = rng.integers(0, 2, n)
    post = rng.integers(0, 2, n)
    y = 1 + 2 * treat + 3 * post + 5 * treat * post + rng.normal(0, 1, n)
    return pd.DataFrame({"y": y, "treat": treat, "post": post,
                         "x1": rng.normal(0, 1, n),
                         "cl": rng.integers(0, 12, n),
                         "w": rng.uniform(0.5, 2.0, n)})


def _staggered(seed=0, n_units=120, n_periods=8):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        g = [0, 4, 6][u % 3]
        fe = rng.normal()
        for t in range(1, n_periods + 1):
            te = 1.0 * (t - g + 1) if (g > 0 and t >= g) else 0.0
            rows.append({"unit": u, "time": t,
                         "y": fe + 0.5 * t + te + rng.normal(0, 0.4),
                         "g": g, "x1": rng.normal(), "cl": u % 15})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def d2x2():
    return _data_2x2()


@pytest.fixture(scope="module")
def stag():
    return _staggered()


# ----------------------------------------------------------------------
# did_2x2 branches (weights / cluster / robust=False / R^2 weighted)
# ----------------------------------------------------------------------

def test_did_2x2_weights(d2x2):
    r = sp.did_2x2(d2x2, y="y", treat="treat", time="post", weights="w")
    assert abs(r.estimate - 5.0) < 0.5
    assert r.se > 0


def test_did_2x2_weights_cluster(d2x2):
    r = sp.did_2x2(d2x2, y="y", treat="treat", time="post",
                   weights="w", cluster="cl")
    assert r.se > 0


def test_did_2x2_weights_robust_false(d2x2):
    r = sp.did_2x2(d2x2, y="y", treat="treat", time="post",
                   weights="w", robust=False)
    assert r.se > 0


def test_did_2x2_cluster_no_weights(d2x2):
    r = sp.did_2x2(d2x2, y="y", treat="treat", time="post", cluster="cl")
    assert r.se > 0


def test_did_2x2_nonbinary_treat_raises(d2x2):
    from statspai.exceptions import MethodIncompatibility
    bad = d2x2.copy()
    bad["treat"] = np.arange(len(bad)) % 3
    with pytest.raises(MethodIncompatibility):
        sp.did_2x2(bad, y="y", treat="treat", time="post")


def test_did_2x2_nonbinary_time_raises(d2x2):
    from statspai.exceptions import MethodIncompatibility
    bad = d2x2.copy()
    bad["post"] = np.arange(len(bad)) % 4
    with pytest.raises(MethodIncompatibility):
        sp.did_2x2(bad, y="y", treat="treat", time="post")


# ----------------------------------------------------------------------
# overlap_weighted_did
# ----------------------------------------------------------------------

def test_overlap_weighted_did(d2x2):
    r = sp.overlap_weighted_did(d2x2, y="y", treat="treat", time="post",
                                covariates=["x1"])
    assert np.isfinite(r.estimate)
    assert r.se > 0


# ----------------------------------------------------------------------
# sun_abraham options
# ----------------------------------------------------------------------

def test_sun_abraham_basic(stag):
    r = sp.sun_abraham(stag, y="y", g="g", t="time", i="unit")
    assert np.isfinite(r.estimate)


def test_sun_abraham_lastcohort_control(stag):
    r = sp.sun_abraham(stag, y="y", g="g", t="time", i="unit",
                       control_group="lastcohort")
    assert np.isfinite(r.estimate)


def test_sun_abraham_bad_control_raises(stag):
    with pytest.raises(ValueError):
        sp.sun_abraham(stag, y="y", g="g", t="time", i="unit",
                       control_group="bogus")


def test_sun_abraham_covariates_cluster(stag):
    r = sp.sun_abraham(stag, y="y", g="g", t="time", i="unit",
                       covariates=["x1"], cluster="cl")
    assert r.se > 0


def test_sun_abraham_event_window(stag):
    r = sp.sun_abraham(stag, y="y", g="g", t="time", i="unit",
                       event_window=(-2, 2))
    assert np.isfinite(r.estimate)


# ----------------------------------------------------------------------
# aggte aggregation types
# ----------------------------------------------------------------------

def test_aggte_types(stag):
    cs = sp.callaway_santanna(stag, y="y", g="g", t="time", i="unit")
    for typ in ("simple", "dynamic", "group", "calendar"):
        agg = sp.aggte(cs, type=typ, bstrap=False)
        assert agg is not None
        assert np.isfinite(agg.estimate)


def test_aggte_bstrap(stag):
    cs = sp.callaway_santanna(stag, y="y", g="g", t="time", i="unit")
    agg = sp.aggte(cs, type="dynamic", bstrap=True, n_boot=50,
                   random_state=1)
    assert agg is not None


def test_aggte_min_max_e(stag):
    cs = sp.callaway_santanna(stag, y="y", g="g", t="time", i="unit")
    agg = sp.aggte(cs, type="dynamic", min_e=-2, max_e=2, bstrap=False)
    assert agg is not None


# ----------------------------------------------------------------------
# lp_did
# ----------------------------------------------------------------------

def test_lp_did(stag):
    df = stag.copy()
    # treatment indicator (1 from first-treat onward)
    df["treated_now"] = ((df["g"] > 0) & (df["time"] >= df["g"])).astype(int)
    r = sp.lp_did(df, y="y", unit="unit", time="time",
                  treatment="treated_now", horizons=(-2, 3))
    assert r is not None


# ----------------------------------------------------------------------
# cohort_anchored_event_study
# ----------------------------------------------------------------------

def test_cohort_anchored(stag):
    r = sp.cohort_anchored_event_study(stag, y="y", treat="g", time="time",
                                       id="unit", leads=3, lags=3)
    assert r is not None


def test_cohort_anchored_cluster(stag):
    r = sp.cohort_anchored_event_study(stag, y="y", treat="g", time="time",
                                       id="unit", leads=2, lags=2,
                                       cluster="cl")
    assert r is not None


# ----------------------------------------------------------------------
# gardner_did
# ----------------------------------------------------------------------

def test_gardner_did(stag):
    r = sp.gardner_did(stag, y="y", group="unit", time="time",
                       first_treat="g")
    assert np.isfinite(r.estimate)


def test_gardner_did_event_study(stag):
    r = sp.gardner_did(stag, y="y", group="unit", time="time",
                       first_treat="g", event_study=True, horizon=(-2, 3))
    assert r is not None


# ----------------------------------------------------------------------
# did_multiplegt_dyn
# ----------------------------------------------------------------------

def test_did_multiplegt_dyn(stag):
    df = stag.copy()
    df["treated_now"] = ((df["g"] > 0) & (df["time"] >= df["g"])).astype(int)
    r = sp.did_multiplegt_dyn(df, y="y", group="unit", time="time",
                              treatment="treated_now", dynamic=2)
    assert r is not None


# ----------------------------------------------------------------------
# ddd_heterogeneous
# ----------------------------------------------------------------------

def test_ddd_heterogeneous():
    rng = np.random.default_rng(2)
    rows = []
    for u in range(120):
        cohort = [0, 4, 6][u % 3]
        sub = u % 2
        fe = rng.normal()
        for t in range(1, 9):
            te = (1.0 * (t - cohort + 1) * (1 + 0.5 * sub)
                  if (cohort > 0 and t >= cohort) else 0.0)
            rows.append({"unit": u, "time": t,
                         "y": fe + 0.3 * t + te + rng.normal(0, 0.4),
                         "cohort": cohort, "sub": sub})
    df = pd.DataFrame(rows)
    r = sp.ddd_heterogeneous(df, y="y", unit="unit", time="time",
                             cohort="cohort", subgroup="sub", n_boot=25)
    assert r is not None


# ----------------------------------------------------------------------
# did_misclassified
# ----------------------------------------------------------------------

def test_did_misclassified(stag):
    # treat column must encode the first-treated cohort (0 = never-treated).
    r = sp.did_misclassified(stag, y="y", treat="g", time="time",
                             id="unit", pi_misclass=0.05)
    assert r is not None


def test_did_misclassified_bad_pi(stag):
    with pytest.raises(ValueError):
        sp.did_misclassified(stag, y="y", treat="g", time="time",
                             id="unit", pi_misclass=0.7)


# ----------------------------------------------------------------------
# continuous_did
# ----------------------------------------------------------------------

def test_continuous_did():
    rng = np.random.default_rng(3)
    rows = []
    for u in range(150):
        dose = rng.uniform(0, 1)
        fe = rng.normal()
        for t in (0, 1):
            te = 2.0 * dose if t == 1 else 0.0
            rows.append({"unit": u, "time": t, "post": t,
                         "y": fe + 0.5 * t + te + rng.normal(0, 0.3),
                         "dose": dose})
    df = pd.DataFrame(rows)
    r = sp.continuous_did(df, y="y", dose="dose", time="time", id="unit",
                          post="post", t_pre=0, t_post=1)
    assert r is not None


# ----------------------------------------------------------------------
# harvest_did
# ----------------------------------------------------------------------

def test_harvest_did(stag):
    r = sp.harvest_did(stag, unit="unit", time="time", outcome="y",
                       cohort="g")
    assert r is not None
