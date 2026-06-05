"""Coverage tests for classic SCM paths: covariates, special predictors,
V-weight methods, conformal inference override, and error branches.

Targets statspai.synth.scm and statspai.synth._core uncovered branches.
"""

import importlib

import numpy as np
import pandas as pd
import pytest

import statspai as sp

scm = importlib.import_module("statspai.synth.scm")


def _panel(n_units=8, n_periods=16, treatment_time=11, effect=4.0, seed=51,
           with_cov=True):
    rng = np.random.default_rng(seed)
    alphas = rng.normal(10, 2, n_units)
    betas = rng.normal(0.5, 0.1, n_units)
    common = rng.normal(0, 0.4, n_periods)
    records = []
    for i in range(n_units):
        for ti, t in enumerate(range(1, n_periods + 1)):
            y = alphas[i] + betas[i] * t + common[ti] + rng.normal(0, 0.2)
            if i == 0 and t >= treatment_time:
                y += effect
            row = {'unit': f'u{i}', 'time': t, 'outcome': y}
            if with_cov:
                row['x1'] = alphas[i] + 0.05 * t + rng.normal(0, 0.1)
                row['x2'] = betas[i] * t + rng.normal(0, 0.1)
            records.append(row)
    return pd.DataFrame(records)


COMMON = dict(outcome='outcome', unit='unit', time='time',
              treated_unit='u0', treatment_time=11)


@pytest.fixture
def panel():
    return _panel()


def test_classic_with_covariates_nested_v(panel):
    """Covariates trigger nested V optimization (_core.solve_synth_weights_adh)."""
    res = sp.synth(panel, **COMMON, method='classic', covariates=['x1', 'x2'],
                   placebo=False)
    assert np.isfinite(res.estimate)
    mi = res.model_info or {}
    # V weights should be populated when predictors come from covariates
    assert 'v_weights' in mi or 'v' in mi or len(mi) > 0


def test_classic_v_method_nested(panel):
    res = sp.synth(panel, **COMMON, method='classic', covariates=['x1'],
                   v_method='nested', n_random_starts=2, placebo=False)
    assert np.isfinite(res.estimate)


def test_classic_v_method_equal(panel):
    res = sp.synth(panel, **COMMON, method='classic', covariates=['x1', 'x2'],
                   v_method='equal', placebo=False)
    assert np.isfinite(res.estimate)


def test_classic_no_standardize(panel):
    res = sp.synth(panel, **COMMON, method='classic', covariates=['x1'],
                   standardize_predictors=False, placebo=False)
    assert np.isfinite(res.estimate)


def test_classic_special_predictors_mean(panel):
    sp_spec = [('x1', [1, 2, 3], 'mean')]
    res = sp.synth(panel, **COMMON, method='classic',
                   special_predictors=sp_spec, placebo=False)
    assert np.isfinite(res.estimate)


def test_classic_special_predictors_sum_and_slice(panel):
    sp_spec = [('x2', slice(1, 5), 'sum')]
    res = sp.synth(panel, **COMMON, method='classic',
                   special_predictors=sp_spec, placebo=False)
    assert np.isfinite(res.estimate)


def test_classic_special_predictor_bad_op(panel):
    sp_spec = [('x1', [1, 2], 'median')]
    with pytest.raises(ValueError):
        sp.synth(panel, **COMMON, method='classic',
                 special_predictors=sp_spec, placebo=False)


def test_classic_special_predictor_missing_col(panel):
    sp_spec = [('nope', [1, 2], 'mean')]
    with pytest.raises(ValueError):
        sp.synth(panel, **COMMON, method='classic',
                 special_predictors=sp_spec, placebo=False)


def test_ridge_default_l2(panel):
    res = sp.synth(panel, **COMMON, method='ridge', placebo=False)
    assert np.isfinite(res.estimate)


def test_inference_conformal_override(panel):
    res = sp.synth(panel, **COMMON, method='classic', inference='conformal',
                   grid_size=15)
    assert np.isfinite(res.estimate)


def test_unknown_backend_raises(panel):
    with pytest.raises(ValueError):
        sp.synth(panel, **COMMON, method='classic', backend='matlab')


def test_r_backend_penalized_not_implemented(panel):
    with pytest.raises(NotImplementedError):
        sp.synth(panel, **COMMON, method='ridge', backend='r')


def test_r_backend_covariates_not_implemented(panel):
    with pytest.raises(NotImplementedError):
        sp.synth(panel, **COMMON, method='classic', covariates=['x1'],
                 backend='synth')


def test_missing_column_raises():
    df = _panel(with_cov=False)
    with pytest.raises(ValueError):
        sp.synth(df, outcome='no_such', unit='unit', time='time',
                 treated_unit='u0', treatment_time=11, method='classic')


def test_treated_unit_not_found_raises(panel):
    with pytest.raises(ValueError):
        sp.synth(panel, **{**COMMON, 'treated_unit': 'ghost'},
                 method='classic')


def test_too_few_pre_periods_raises():
    df = _panel(with_cov=False)
    with pytest.raises(ValueError):
        # treatment_time=2 leaves only 1 pre-period
        sp.synth(df, outcome='outcome', unit='unit', time='time',
                 treated_unit='u0', treatment_time=2, method='classic')


def test_no_post_periods_raises():
    df = _panel(with_cov=False, n_periods=16)
    with pytest.raises(ValueError):
        sp.synth(df, outcome='outcome', unit='unit', time='time',
                 treated_unit='u0', treatment_time=100, method='classic')


def test_classic_placebo_full(panel):
    res = sp.synth(panel, **COMMON, method='classic', covariates=['x1'],
                   placebo=True)
    mi = res.model_info or {}
    # placebo should populate gap distributions / rmspe ratios
    assert any(k in mi for k in
               ('placebo_gaps', 'placebo_effects', 'rmspe_ratios',
                'placebo_units', 'pvalue'))
