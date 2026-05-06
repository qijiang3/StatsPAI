# Paper-AgentBench — Companion paper workspace

This directory is the working home for the **CausalAgentBench** companion
benchmark paper, a planned follow-up to the StatsPAI JSS submission. It
exists so that material we deliberately removed from the JSS draft
(the production RCT for LLM-agent behavioural evaluation, the 900-trial
$3 \times 2$ factorial design, the OSF pre-registration protocol, the
mock-LLM dry-run results) is **preserved verbatim**, not deleted.

The split is the original two-paper plan recorded in
`Paper-JSS/JSS-research-plan.md`: JSS publishes the unified-package +
parity story; the companion paper publishes the agent-behavioural RCT.
Splitting them into separate manuscripts at the JSS revision stage
preserves the pre-registration's scientific value (CausalAgentBench is
deposited on OSF before its trials run) and lets the JSS paper close
under a single, falsifiable scope.

## Layout

```
Paper-AgentBench/
├── README.md                                       (this file)
├── archive-from-jss/                               (preserved verbatim)
│   ├── jss-section7-agent-eval-original.tex        (JSS §7, 130 lines, pre-trim)
│   └── jss-context-around-causalagentbench.tex     (§1.3 item 4 + §9 mentions)
└── manuscript/                                     (companion-paper draft home)
    ├── sections/                                   (empty; populated when drafting begins)
    └── notes/
        └── osf-preregistration.md                  (copy of Paper-JSS/notes/...)
```

## Status (2026-05-05)

- **Archive**: complete. Every passage cut from the JSS draft during the
  P0-3 reviewer-response trim is preserved under `archive-from-jss/`.
- **Manuscript**: not yet drafted. The companion paper is gated on
  (i) OSF pre-registration deposit, (ii) production-API budget approval,
  (iii) JSS submission of the parent paper.
- **Pre-registration protocol**: see
  `manuscript/notes/osf-preregistration.md` (also kept in
  `Paper-JSS/notes/` for legacy paths).

## Working principles

1. **Never delete archive content.** If the companion paper's draft
   diverges from the archive, leave the archive intact and reword in
   `manuscript/sections/` instead.
2. **JSS draft references the companion paper as a forward pointer.**
   The JSS §7 / §1.3 / §9 passages now redirect readers here rather
   than promising results that have not yet been collected.
3. **OSF pre-registration is the canonical protocol.** Any future
   methodological change to CausalAgentBench is tracked through OSF
   amendments, with `manuscript/notes/osf-preregistration.md` as the
   working copy.
