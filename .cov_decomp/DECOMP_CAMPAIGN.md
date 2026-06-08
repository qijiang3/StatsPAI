# decomposition-module ≥95% coverage track (parallel to core campaign)

**Scope:** the `decomposition` submodule — Oaxaca / RIF / FFL / DFL / CFM /
Machado–Mata / Melly / Kitagawa / inequality / Yu–Elwert / causal decompositions.
This is **NOT** one of the six core campaign modules (`did iv rd synth dml panel`,
owned by the `.coverage_campaign/` owner agent). This track exists to take an
**unclaimed** high-value module to ≥95% without colliding with that campaign.

**Owner:** this agent (assigned by maintainer 2026-06-05).
**Budget:** ~3 months. **Quality bar:** real numerical/identity assertions, no
mocking (CLAUDE.md §5/§12).

## Conflict-avoidance rules (same spirit as the core campaign)
- Test-only. New files named `tests/test_decomposition_cov_<topic>.py`.
- Never touch `src/`, the core campaign's `.coverage_campaign/`, or its files.
- Scratch lives in `.cov_decomp/` (this dir) with a private `COVERAGE_FILE`.
- `git fetch --rebase` before each push; push test-only commits direct to main.

## Measurement
Sub-package `--cov=statspai.decomposition` CRASHES (scipy highspy double-register,
same ObjSense bug noted in the core campaign). Always whole-package:
```
COVERAGE_FILE=.cov_decomp/.cov pytest tests/test_decomposition_cov_*.py \
  -o addopts="" --cov=statspai --cov-report=term-missing
```
Incremental gap-closure is verified by line-level monotonicity (a baseline-
uncovered line that my file hits is closed in the full suite too).

## Baseline (from core campaign's owner_state.xml, 2026-06-05, full-suite)
decomposition: **71.6%** (2143/2994), ~851 lines to 95%.

| file | baseline cov% | uncov | status |
|---|---|---|---|
| oaxaca.py | 65.3 | 115 | ✅ ~99% (test_decomposition_cov_oaxaca.py, 25 tests) |
| plots.py | 26.2 | 175 | ⬜ |
| _common.py | 68.4 | 115 | ✅ ~79%+ my-file (test_decomposition_cov_common.py, 29 tests) |
| causal.py | 72.2 | 73 | ⬜ |
| inequality.py | 76.9 | 71 | ✅ ~87%+ my-file (test_decomposition_cov_inequality.py, 22 tests) |
| machado_mata.py | 67.5 | 51 | ⬜ |
| yu_elwert.py | 76.3 | 49 | ⬜ |
| ffl.py | 71.6 | 42 | ⬜ |
| _results.py | 81.0 | 32 | ⬜ |
| cfm.py | 79.5 | 25 | ⬜ |
| kitagawa.py | 82.3 | 23 | ⬜ |
| nonlinear.py | 86.5 | 22 | ⬜ |
| dfl.py | 85.3 | 21 | ⬜ |
| rif.py | 85.3 | 21 | ⬜ |
| melly.py | 78.4 | 16 | ⬜ |

## Session log
### 2026-06-05 — session 1
- Withdrew from core campaign (owner already had iv/dml ≥95%, overlap). Took
  decomposition as the unclaimed module per maintainer.
- oaxaca.py 65.3% → ~99% (my-file-only). 25 tests, all real algebraic
  identities: Oaxaca two-fold additivity for every reference weighting
  {0,1,pooled,cotton,reimers}; reference→beta* selection; detailed sums to
  aggregate; Gelbach exact additivity (total_change == base−full coef,
  sum(delta)==total_change); full rendering surface + every validation branch.
  Residual 4 lines (198, 398-399, 869) are defensive/edge.
