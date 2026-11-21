# Source repair handoff — 2026-11-21

## Inputs and merge

Initial input: edf50b3e961823e6110fa4472590ed1573f957a2.
Concurrent additions merged through 100c6dba417a54bfa2a36ff7f2d1c3826f2a4779:
hot/cold query skew/probes, new-fact targets, controlled geometry script and
their regression checks. The existing joint recurrence is preserved.

## Changes

Pure helpers expose covariance timing, Cholesky solves and the actual state
step. MATH.md derives them from one strictly convex objective. Recurrent math
uses FP32 (explicit FP64 retained). Geometry scaling and statistics gradients
are explicit options. The one content state still needs a K-by-K covariance.

The runner enables a causal convolution in every arm. This is an architecture
change, not an exact replay of old results. Relevant settings are recorded.

Evaluation preserves hot/cold and genuinely new associations, expands held-out
sampling, records counts and paired per-sequence data, and uses a conservative
developmental screen. A hot-value-only strategy is a necessary comparison;
vocabulary-uniform chance is inadequate. Whole requested matrices must finish.
No scientific GO/PASS is generated.

## Verification state

Completed: primary papers/author-code inspection, independent algebra review,
static source/interface review and Python AST syntax parsing.

Authored but unexecuted: revised unit tests and controlled numerical checks.
Pending: target runtime, CUDA behavior, training, native benchmark reproduction,
measured throughput and scientific efficacy.

Incoming documentation reported earlier CPU checks. Those historical statements
do not validate this repaired revision. Follow LOCAL_AGENT_RUNBOOK.md and return
raw evidence tied to the final commit. The Web reviewer did not import project
modules, execute tests/forwards or launch training, following the research-autopilot
source-review role; this does not mean execution is unnecessary.
