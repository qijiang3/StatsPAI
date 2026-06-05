# Core-module ≥95% coverage campaign

**Goal (set by maintainer, 2026-06-05):** drive all six core estimator modules
— `did iv rd synth dml panel` — to a **hard ≥95% line coverage**, measured under
the *full test suite*. Budget: ~3 months. Quality bar is non-negotiable
(CLAUDE.md §5): real numerical assertions, parity/analytic tests, **no mocking of
numerical paths**.

This file is the durable, cross-session tracker. The maintainer verifies
progress against the **Acceptance checklist** at the bottom.

---

## How coverage is measured here (important)

StatsPAI coverage can only be measured with the **whole-package** source:

```bash
pytest tests/ -q --cov-report=xml:.coverage_campaign/latest.xml --cov-report=
python scripts/coverage_campaign.py report --xml .coverage_campaign/latest.xml
```

- `--cov=statspai.<sub>` (sub-package source) **crashes** with
  `generic_type: type "ObjSense" is already registered` — it reorders imports
  ahead of the `scipy.optimize` PyO3/pybind11 stabiliser in `tests/conftest.py`.
  Always use the default whole-package `--cov=statspai`.
- The authoritative number is the **full-suite** number. Running only
  `tests/test_<mod>*.py` understates a module badly (iv: 38% module-only vs
  87% full-suite) because cross-module tests exercise it too.
- `scripts/coverage_campaign.py gaps <mod> --per-file` lists the exact uncovered
  line ranges to target.

---

## Baselines (committed `coverage.xml`, 2026-05-28, v1.16.0)

| module | cov% | covered/total | +lines to 95% |
|---|---|---|---|
| iv | 86.7 | 2426/2799 | 234 |
| dml | 75.7 | 959/1267 | 245 |
| panel | 54.0 | 970/1796 | 737 |
| did | 74.6 | 4369/5859 | 1198 |
| rd | 66.3 | 3107/4689 | 1348 |
| synth | 66.0 | 3922/5940 | 1721 |
| **total** | ~70.5 | | **≈5,480** |

Sequencing (cheapest first, big three last): **iv → dml → panel → did → rd → synth**.

---

## Hard rules for this campaign

1. **Test-only.** Add tests under `tests/`. Do **not** change estimator
   numerics — that could move JOSS/JSS dossier numbers (review #10604).
2. **If a test reveals a real bug:** stop, report to maintainer, then follow
   CLAUDE.md §12 `⚠️ correctness fix` (CHANGELOG + MIGRATION). Never silently
   change numerical output during JOSS review.
3. **Do not touch** the JOSS/JSS files in flight: `docs/joss_validation_dossier.md`,
   `tests/test_jss_validation_api.py`, `tests/external_parity/*`,
   `tests/reference_parity/_fixtures/dml_iv_*`.
4. **Conflict avoidance with parallel agents.** Hot src areas observed:
   did-imputation (`codex/bjs-imputation-fix`), panel xtabond/xtnbreg
   (`fix/xtabond-instruments`, `codex/xtnbreg-panel-nbreg`),
   `worktree-improve-correctness`, `parity-stabilization`. Write **new** test
   files named `test_<mod>_cov_<topic>.py`; `git pull --rebase` before each
   push; schedule hot src files last and re-check `git log` before starting them.
5. New tests must carry real assertions (closed-form / parity / invariants),
   not just smoke-call for coverage.

---

## Per-module status

| module | start | current | target | status |
|---|---|---|---|---|
| iv | 86.7 | — | 95 | 🟡 in progress |
| dml | 75.7 | — | 95 | ⬜ queued |
| panel | 54.0 | — | 95 | ⬜ queued |
| did | 74.6 | — | 95 | ⬜ queued |
| rd | 66.3 | — | 95 | ⬜ queued |
| synth | 66.0 | — | 95 | ⬜ queued |

---

## Session log

### 2026-06-05 — session 1: infrastructure + iv start
- Diagnosed local coverage blocker (sub-package `--cov` crashes on scipy highspy
  double-registration); established whole-package measurement path.
- Added `scripts/coverage_campaign.py` (per-module report + gap line ranges).
- Kicked off authoritative full-suite baseline → `.coverage_campaign/`.
- Started module 1: **iv**. Added `tests/test_iv_cov_dispatcher_routes.py`
  (9 tests, green): covers the previously-untested `sp.iv(method=...)` routes
  `jive_mw, many_weak_ar, lasso, post_lasso, mte, ivmte_bounds,
  plausibly_exog_uci/ltz` and, transitively, `many_weak.py / post_lasso.py /
  plausibly_exogenous.py / mte.py / ivmte_lp.py`. Each asserts a real property
  (point estimate near the DGP truth, or a valid CI/bound ordering), not a smoke
  call. Proven loop: gap-map → targeted tests → green.
- **Pending decision from maintainer:** commit/push cadence (see report).

### 2026-06-05 — handoff from parallel JOSS-prep agent

- A second agent working the JOSS-prep track contributed one **synth**
  coverage file before withdrawing from the coverage campaign (maintainer
  decision: campaign owns all of W3.2). Handed over, renamed to the campaign
  convention: `tests/test_synth_cov_plots.py` (12 tests, green) — covers the
  `sp.synthplot` dispatcher's plotting layer (`synth/plots.py`): types
  `trajectory, gap, both, weights, placebo, placebo_gap, rmspe, conformal,
  compare`, the `pre_band` overlay, and the unknown-type `ValueError`. These
  are rendering smoke tests (assert a Matplotlib `Figure` is returned), so
  they lift `synth/plots.py` (1,237 lines, 23% baseline) without pinning
  pixels. Remaining synth plot renderers needing specialised fits
  (`staggered, factors, distributional, multi_outcome, prediction_interval,
  sensitivity`) are still open for the campaign's synth module.

---

## Acceptance checklist (for the maintainer to verify all results)

Run, then confirm each line:

```bash
# 1. fresh authoritative full-suite coverage
pytest tests/ -q --cov-report=xml:.coverage_campaign/verify.xml --cov-report=
python scripts/coverage_campaign.py report --xml .coverage_campaign/verify.xml
```

- [ ] iv ≥ 95%
- [ ] dml ≥ 95%
- [ ] panel ≥ 95%
- [ ] did ≥ 95%
- [ ] rd ≥ 95%
- [ ] synth ≥ 95%
- [ ] `pytest tests/ -q` fully green (no new failures)
- [ ] `pytest tests/reference_parity/ -q` still passes (numbers unmoved)
- [ ] no estimator numerics changed (git diff over `src/statspai/{did,iv,rd,synth,dml,panel}` is test-enabling only, or each change is a logged ⚠️ correctness fix)
- [ ] CI per-module coverage ratchet in place (no silent regressions)
