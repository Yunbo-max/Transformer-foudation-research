# Source repair handoff — updated 2026-11-22

## Follow-up after method review

Parent revision: b4c0b373759f31be4963f56f850dae1d0c3076a6.
The predictor, recurrence, default gates and all control arms are retained.
This follow-up adds event-conditioned diagnostic sums/counts and fixes the
description of the immediate alias probe. No delayed-alias condition is added.
Generation consumes the same random draws and produces the same token stream;
new metadata labels existing value-write positions only. The detailed event
report is opt-in via --diagnostics and never enters model inputs or gating.

METHOD_ZH.md explains the motivation, current-key versus previous-query
feedback and the inherited versus integrated components. SOURCES.md makes the
scalar/channel-wise and gate-memory gradient differences explicit.
New diagnostic regression checks are authored for local execution. All runtime
and scientific evidence below remains pending for the revised source hashes.

## Inputs and attribution

Initial input: 6e1e7f0184bbe6af3ba070edd06a49d00b40edae.
Concurrent protocol updates merged through 8e47e35b4e96e72494b516d509b2b90489743cd9,
including hot/cold probes, genuinely new facts and tests.

Vuk Rosic's pinned code already supports --lpsE 1, scaling both erase and write.
The earlier novelty claim for joint suppression is corrected.

## Changes

Reference and official paths now share raw/native guard policies, single
normalization boundaries and FP32 state. The independent PyTorch guard follows
the author certificate and straight-through derivative. Exact joint active-edit
interpolation is claimed only for raw gates; projection is not homogeneous.

Per-head gamma defaults to .25. All arms use causal QKV convolution by default.
Both choices change the prior configuration; old checkpoints/results cannot be
silently relabeled. The cosine predictor remains a heuristic. New-key/same-value
errors, magnitude blindness, stale reads and decay/refresh tension are documented,
not assumed solved.

Official training requires a matching forward/state/gradient receipt binding core
and verifier hashes, pinned upstream code, actual FLA normalization, runtime,
hardware, precision, guard and configuration. Changed settings need new evidence.

The common runner preserves concurrent hot/cold and new-fact protocol updates,
saves more held-out observations and paired uncertainty, and requires all
requested controls to finish. Diagnostic promise is not a scientific PASS.

## Verification state

Completed: primary author-code review, algebra and independent source/interface
review, static AST syntax checks. Authored but unexecuted: revised unit tests and
CUDA parity checks. Pending: target-machine execution, parity, training, native
benchmarks and measured cost/benefit.

Historical smoke statements do not validate these modified files. The Web
reviewer did not import project modules, execute forwards/tests, install
dependencies or launch training, following the research-autopilot source-review
role. Use LOCAL_AGENT_RUNBOOK.md and return raw results tied to the final commit.
