# OSF Pre-registration — StatsPAI JSS Tracks B and D

> Template for the Open Science Framework pre-registration that the
> JSS plan calls for at the end of Phase A. Deposit this document on
> OSF before any benchmark trial is executed; the OSF DOI is then
> cited from the JSS manuscript §5 (Track B) and §7 (Track D).

---

## Title

Pre-registration of statistical-validity and behavioural-evaluation
benchmarks for the StatsPAI Python platform.

## Authors

Biaoyue (Bryce) Wang. CoPaper.AI; Stanford REAP, Stanford University.
brycew6m@stanford.edu  ·  ORCID 0000-0002-1828-2208

## Date and version

[Date: deposit before Phase B execution begins, target 2026-08-15]
Version 1.0

## Summary

This pre-registration freezes the protocols for two benchmark
exercises associated with the StatsPAI JSS submission:

- **Track B — Statistical validity by Monte Carlo coverage.** Eight
  estimators evaluated on known data-generating processes, with
  pre-registered tolerance budgets for bias and CI coverage.
- **Track D — Behavioural evaluation of LLM agents
  (CausalAgentBench).** 50 prompts × 6 conditions × 3 reps = 900
  trials of LLM agents performing causal-inference tasks, with
  pre-registered hypothesis tests on five primary endpoints.

The protocols are frozen prior to any data collection. Any deviation
required during execution will be recorded as a revision on this
OSF entry and disclosed in the JSS manuscript as a deviation rather
than incorporated silently.

## Track B — Monte Carlo coverage

### Estimators

The eight estimators evaluated are: 2SLS (single endogenous
variable), Callaway--Sant'Anna staggered DiD, Sun--Abraham event
study, RD bias-corrected (CCT 2014), Synthetic DID (Arkhangelsky et
al. 2021), DML PLR (Chernozhukov et al. 2018), causal forest (Wager
& Athey 2018), and classical synthetic control (Abadie, Diamond &
Hainmueller 2010).

### Data-generating processes

Each estimator runs on two DGPs: a baseline DGP under which all
identification assumptions hold, and a robustness DGP that violates
one specific assumption (weak instrument; staggered timing with
heterogeneous effects; bandwidth at the boundary of the bias-variance
trade-off; placebo treatment date for SDID; etc.). The exact
functional forms of the DGPs are listed in
`Paper-JSS/experiments/track-b-monte-carlo/dgp/<estimator>.py` of
the replication archive at the SHA recorded in this OSF deposit.

### Sample sizes and replications

- $N \in \{500, 5{,}000\}$ per replication for $T \le 30$ panel DGPs.
- $T = 5$ for the staggered-DiD DGP; $T = 30$ for the synthetic
  control DGP.
- 1{,}000 Monte Carlo replications per (estimator, DGP, $N$) cell.
- A separate fast variant with 100 replications for sanity checking.

### Pre-registered metrics

For each (estimator, DGP, $N$) cell:

- Bias: mean of $(\hat\theta - \theta_0)$ across replications.
- RMSE: $\sqrt{\mathrm{mean}((\hat\theta - \theta_0)^2)}$.
- Empirical 95% CI coverage: share of replications whose 95% CI
  contains $\theta_0$.
- Average CI half-width.

### Pre-registered tolerance budget

- $|\mathrm{bias}| < 0.01\,\sigma_y$ on the baseline DGP.
- Empirical 95% coverage in $[0.93, 0.97]$ on the baseline DGP
  (Wilson interval at $B = 1{,}000$).
- Robustness DGPs: bias and coverage are reported but not held to
  the baseline tolerance; the purpose of the robustness DGPs is to
  characterise the failure mode, not to certify a tolerance.

### Pass/fail rules

- An estimator that fails the bias or coverage tolerance on the
  baseline DGP is reported as a "Track B fail" in the JSS paper,
  with the failure mode discussed in §5.6. We do **not** drop
  failing estimators from the paper.
- The simple-ATT aggregation in Callaway--Sant'Anna is known at
  pre-registration time to under-cover (see
  `tests/coverage_monte_carlo/FINDINGS.md` of the StatsPAI
  repository at this SHA); we expect this to remain a Track B fail
  unless the v1.14 multiplier-bootstrap fix lands before benchmark
  execution.

## Track D — CausalAgentBench

### Task set

Fifty causal-inference research prompts, each pairing a research
question with a sandboxed CSV file. Distribution by difficulty:

- L1 (direct): 20 prompts. Method is named in the prompt.
- L2 (indirect): 20 prompts. Identification structure is described;
  agent picks the method.
- L3 (workflow): 10 prompts. Full workflow including diagnostics
  and robustness.

The 50 prompts, gold-answer point estimates, and grading rubrics are
deposited under
`Paper-JSS/experiments/track-d-agent/prompts/` and
`.../golds/` of the replication archive at the SHA recorded in this
OSF deposit. The prompts are released under CC-BY-4.0; the
gold-answer files are released under MIT.

### Conditions

A 3 × 2 factorial design:

|             | Anthropic Claude (specific model and version recorded at trial 1) | OpenAI GPT (specific model and version recorded at trial 1) |
|---|---|---|
| StatsPAI + MCP            | C1 | C2 |
| Pythonic stack            | C3 | C4 |
| R via MCP                 | C5 | C6 |

The Pythonic stack consists of statsmodels, linearmodels, DoubleML
(Python), grf-python, EconML, CausalML, and standard SciPy data
manipulation, with no StatsPAI surface available. The R-via-MCP
condition uses radian + Jupyter R kernel exposed to the agent
through an MCP shim, with packages MatchIt, did, fixest, rdrobust,
Synth, synthdid, DoubleML, grf, and HonestDiD installed.

Each cell runs the 50 prompts with three independent random seeds,
giving 150 trials per cell and 900 trials in total. Sampling is
\code{temperature = 0} for reproducibility; randomness across reps
is seeded only by the data-loading order and any non-deterministic
agent-action seeds, which are recorded.

### Pre-registered metrics

Primary:

- M1. Task success: share of trials with final point estimate within
  $\pm 5\%$ of gold answer.
- M2. Method correctness: share of trials with the gold-rubric-correct
  estimator (LLM-as-judge plus 20\% human cross-check).
- M3. Code-execution success: share of trials whose code runs to
  completion without an unhandled exception.
- M4. Token efficiency: median total input + output tokens per trial.

Secondary:

- M5. Hallucination rate: share of trials calling a non-existent
  function (machine-checkable for C1/C2/C3/C4; manual review for
  C5/C6).
- M6. Diagnostic completeness: share of required diagnostic checks
  the agent reports.
- M7. Reproducibility: across-rep variance of the final estimate
  within a (prompt, condition) cell.
- M8. Time-to-result: median wall-clock per trial.

### Pre-registered hypotheses

- **H1.** \statspai{} conditions (C1, C2) achieve task success
  $\geq 90\%$ on L1, $\geq 70\%$ on L2, $\geq 50\%$ on L3.
- **H2.** \statspai{} conditions exceed Pythonic-stack conditions
  (C3, C4) on L2-L3 task success by at least 15 percentage points.
- **H3.** \statspai{} conditions have hallucination rate $< 5\%$
  across all trials, while Pythonic-stack conditions exceed 15\%.
- **H4.** \statspai{} conditions use $\leq 60\%$ of the tokens used
  by Pythonic-stack conditions on the same prompt and language model.
- **H5.** R-via-MCP conditions (C5, C6) have task-success rates
  not statistically distinguishable from \statspai{}, but require
  at least 1.5$\times$ as many tokens.

### Statistical test

- Cluster bootstrap with the prompt as cluster, $B = 9{,}999$
  bootstrap reps.
- Two-sided $\alpha = 0.05$.
- Bonferroni correction across the 5 primary hypotheses
  ($\alpha_{\mathrm{adj}} = 0.01$).
- Effect-size reports use difference-in-shares with
  bootstrap-percentile 95% CIs.

### Stop rules and deviations

- If H3 (hallucination rate) shows the StatsPAI condition exceeds
  5\%, this is reported as a no-go signal for the agent-native claim
  and discussed in §7.4 of the JSS paper. We do not modify the
  StatsPAI surface in response to a Track D failure during the
  pre-registered run.
- API rate limits: if any condition is rate-limited mid-run, the
  affected trials are marked as "rate-limited" rather than
  re-attempted with different seeds, to avoid post-hoc selection.
- Cost overrun: budget cap of \$1{,}000 USD across all conditions.
  If the cap is reached before all trials complete, the protocol is
  paused, the partial data is reported with the
  remaining-trial deficit, and no inference is drawn from the
  paused state.

## Replication archive

The full protocols, datasets, and code are deposited at
[https://github.com/brycewang-stanford/StatsPAI-JSS-replication]
(SHA recorded in this OSF deposit). Both the StatsPAI package and the
benchmark code are released under the MIT licence.

## Authorship and contributions

Sole author: Biaoyue (Bryce) Wang.

Acknowledgement: portions of the benchmark harness code and the
present pre-registration document were prepared with assistance from
a large language model. All experimental decisions, statistical
choices, and numerical results are authored, reviewed, and verified
by the human author.
