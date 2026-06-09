"""Tier D guard — pin the LaLonde classic-track replication numbers.

Part of the P1 campaign (see ``.tierd_campaign/CAMPAIGN.md``). The
``sp.replicate('lalonde_1986')`` registry ships *golden numbers* for the classic
track, but no test guarded the live 1:1 NN propensity-score-matching ATT — so a
tie-handling change in ``sp.match`` silently drifted it from the originally
pinned $2,012.5 to the current deterministic $1,963.4 (a 2.5% move) without any
red flag. This test is that missing guard: it pins all three classic LaLonde
numbers to their current deterministic values on the bundled real data, so any
future drift fails loudly.

The naive and covariate-adjusted OLS values reproduce R ``MatchIt`` to the
dollar; the PSM value is the current deterministic ``sp.match(method='nearest')``
output (sensitive to tie-breaking on the binary covariates, hence the value the
registry pin was refreshed to).

Purely additive — no estimator numerics changed (campaign red line).
"""

import pytest

import statspai as sp

COVS = ["age", "educ", "black", "hispanic", "married", "nodegree", "re74", "re75"]


@pytest.fixture(scope="module")
def lalonde():
    df, _ = sp.replicate("lalonde_1986")
    return df


def test_naive_ols_matches_matchit(lalonde):
    naive = sp.regress("re78 ~ treat", data=lalonde, robust="hc1")
    assert float(naive.params["treat"]) == pytest.approx(-635.03, abs=1.0)


def test_adjusted_ols_matches_matchit(lalonde):
    adj = sp.regress("re78 ~ treat + " + " + ".join(COVS), data=lalonde, robust="hc1")
    assert float(adj.params["treat"]) == pytest.approx(1548.24, abs=1.0)


def test_psm_att_is_deterministic_and_pinned(lalonde):
    # Deterministic across runs; pinned to the current value so any tie-break /
    # algorithm change in sp.match is caught (the guard that was missing before).
    vals = [
        float(
            sp.match(
                data=lalonde, y="re78", treat="treat", covariates=COVS, method="nearest"
            ).estimate
        )
        for _ in range(3)
    ]
    assert len(set(round(v, 6) for v in vals)) == 1  # deterministic
    assert vals[0] == pytest.approx(1963.43, abs=1.0)


def test_psm_recovers_experimental_benchmark(lalonde):
    # The scientific content of DW (1999): matching recovers the ~$1,794
    # experimental benchmark and removes the negative naive selection bias.
    naive = float(
        sp.regress("re78 ~ treat", data=lalonde, robust="hc1").params["treat"]
    )
    psm = float(
        sp.match(
            data=lalonde, y="re78", treat="treat", covariates=COVS, method="nearest"
        ).estimate
    )
    assert naive < 0 < psm
    assert 1500.0 < psm < 2500.0
