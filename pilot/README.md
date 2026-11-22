# Predicted-Skip on channel-wise GDN-2

**Current revision: source-reviewed repair; tests, CUDA parity and training on
these exact bytes are pending local execution.**

Start with [LOCAL_AGENT_RUNBOOK.md](LOCAL_AGENT_RUNBOOK.md).
For the method's purpose and the GDN-2 comparison in Chinese, read
[METHOD_ZH.md](METHOD_ZH.md).
See [MATH.md](MATH.md) for the equations, guard semantics and counterexamples,
[SOURCES.md](SOURCES.md) for pinned author implementations, and
[WEB_HANDOFF.md](WEB_HANDOFF.md) for changed settings and pending checks.

## What the added feedback is for

Predicted-Skip tests whether some retrieved answers are written back into
memory unnecessarily when they re-enter the causal token stream. The added
signal compares the current value with a previous-query read from the
chunk-start state, then softly reduces the active edit.

GDN-2 already has a state-dependent erase term and learned input gates. Its
current-key update and this previous-query signal answer different questions.
The latter may identify answer writeback even when the current write key
differs from the retrieval query. The same distinction can also suppress a
useful new association, so retaining new-key learning is part of the test.
GDN-2 is a base on which to evaluate this hypothesis; it does not require
Predicted-Skip by construction.

## Attribution and scope

Vuk Rosic's original Predicted-Skip code already includes **both write-only
suppression and joint erase/write suppression**, through --lpsE 1. Both are
credited to that work. The earlier claim that joint suppression itself was
new to this branch is withdrawn.

This project adapts those controls to the exact channel-wise GDN-2 recurrence
and evaluates numerical correctness, retention/plasticity and cost. It is
independent of SG-Delta. An implementation extension is not yet a new
paper-level method or a claim of superior model quality.

## Recurrence and predictor

For key-major S with shape K x V:

    Sbar_t = diag(exp(g_t)) S_{t-1}
    S_raw,t = Sbar_t - k_t [(b_t * k_t)^T Sbar_t] + k_t (w_t * v_t)^T
    r_t = S_chunk_start^T q_{t-1}
    rho_t = exp(-gamma_head * max(0, cosine(r_t, v_t)))

write: scale only w by rho.
joint: scale both b and w by rho.

For the RAW recurrence:

    S_joint,t = Sbar_t + rho_t (S_raw,t - Sbar_t)

Decay is still active. A high value cosine is not a certificate that the
current key/value association is already present. New-key/same-value binding,
weak same-direction reads and continual decay are explicit counterexamples.
The boundary-state prediction omits within-chunk updates; the first chunk's
gate is one. No labels or phase masks enter the predictor.

## Numeric and architecture repair

- Per-head learnable gamma follows the author's parameterization. This
  pilot selects 0.25 as its initial strength; the author's --lps is configurable
  and defaults to disabled. --gamma records alternative initializations.
- Causal Q/K/V depthwise short convolution size 4 is applied to all arms.
  --conv-size 0 restores the former no-convolution ablation.
- Q/K normalization is defined consistently and official inputs are not
  accidentally normalized twice.
- Reference state accumulates in FP32. The standalone raw algebra helper
  preserves explicit FP64 inputs for mathematical unit checks.
- --precision float32 or bfloat16 controls the SAME path in validation,
  training and evaluation.
- Default --guard native implements the NVIDIA beta-contrast projection in
  both backends, including its straight-through backward convention.
- --guard raw selects the original unguarded recurrence. Its exact Joint
  interpolation identity is tested separately.
- The default native guard runs AFTER skip scaling, matching NVIDIA behavior.
  Because guard(rho*b) generally differs from rho*guard(b), raw interpolation
  is not claimed for independently guarded arms.
- --differentiate-gate-memory removes the default stop-gradient at the
  predictor's boundary-state input. The main recurrent gradient is retained
  in either case.
- --diagnostics records actual-trajectory active-edit quantities during
  reference evaluation only. Event summaries distinguish initial facts,
  alias facts, genuinely new facts and each answer group. They record sums
  and token/head counts before computing means, and do not alter the predictor.

These settings change the training configuration and gamma parameter shape.
Old scalar-gamma checkpoints are not silently reinterpreted as new checkpoints.

## Local reference commands

    python3 -m unittest discover -s pilot -p 'test_*.py'
    python3 pilot/oracle_diagnostic.py
    python3 pilot/run.py --profile smoke --methods baseline joint --seeds 0 --eval-batches 1 --eval-batch-size 2 --budget-seconds 120
    bash pilot/run_3h.sh --device cuda --backend reference --guard native --precision float32 --seeds 0 1

A one-seed smoke run is INCONCLUSIVE and is not a method result. Runs use new
timestamped subdirectories under runs/predskip_joint/ and never overwrite
earlier result directories.

## Official kernel path

Use the upstream installation instructions at this exact revision; no NVIDIA
source is vendored.

    git clone https://github.com/NVlabs/GatedDeltaNet-2.git
    git -C GatedDeltaNet-2 checkout de8476de0fecae08fc16f0bf3154900244c61c73

After installing its documented dependencies in the target environment:

    python3 pilot/validate_official.py --official-root ./GatedDeltaNet-2 --precision float32 --guards raw native --hidden 128 --heads 2 --key-dim 16 --value-dim 16 --chunk 16 --conv-size 4 --gamma 0.25 --output runs/predskip_joint/parity.json
    bash pilot/run_3h.sh --device cuda --backend official --official-root ./GatedDeltaNet-2 --guard native --precision float32 --parity-report runs/predskip_joint/parity.json --seeds 0 1

The runner requires a successful report matching current source hash, pinned
clean upstream checkout, precision, guard, methods, shape and gate settings.
A changed configuration needs a new report. Do not relax tolerances to make a
failed comparison pass. Validation compares outputs, final states, all
operator input gradients, and model parameters rather than only q_proj.

This remains a small matched-projection decoder, not NVIDIA's full 1.3B model.
Every arm uses the same external chunk dispatch. Boundary-state reads and
repeated kernel launches have a cost; soft skip does not itself skip compute.
No GPU parity or speed result is claimed for the generated repair.

## Inspecting what gets suppressed

With --backend reference --diagnostics, each evaluation panel's JSON includes
gate_events.layer0 (and one entry for each additional layer). The event names
include initial_fact, alias_fact, new_fact, answer_alias and answer_new_post.
Each entry contains token_count, head_token_count, rho_below_half_count and
metrics with sum, count and mean. Available measurements include rho, cosine,
read/value norms, unskipped/applied edit norms and guard adjustment.

The fact events mark the value token in [FACT, key, value]. Answer events
mark the answer token's own write, one position after the logit used to score
that answer. Loss/group masks are shifted explicitly for this purpose.
The labels are evaluator metadata and are never passed to the mixer.

Zero-count events are absent. Old-post and its hot/cold subgroups overlap;
do not add their counts together. The panel combines sums and counts, rather
than averaging unequal batch means. These are event-conditioned observations,
not labels proving that a particular write is necessary or redundant. Short
convolution can carry an association into subsequent token representations.
The current alias is queried immediately; no delayed-alias task is implemented.

## Evaluation and sizing

Arms: baseline, write, joint, shuffled, keyread. The inherited generated
associative-recall diagnostic is not official MQAR, FineWeb or RULER.
Old-fact retention, new alias-key learning and new_post learning are all
recorded. Default evaluation uses 256 independent sequences per panel,
with identical samples across arms, raw per-sequence outcomes and exact counts.

The report requires two distinct completed training seeds, old recall above the
contextual-value baseline, complete matched samples, paired uncertainty and preserved
plasticity. It outputs only INCONCLUSIVE, NO_CLEAR_DEVELOPMENTAL_ADVANTAGE or
PROMISING_DEVELOPMENTAL, never a native benchmark or paper success.

Small profiles are smoke D32/L1/H1/K8, pilot D128/L2/H2/K16, and stress
D192/L3/H3/K32. Counts now print from the actual selected configuration.

    python3 pilot/run.py --profile 100m-sizing --methods joint --dry-run
    python3 pilot/run.py --profile 340m-sizing --methods joint --dry-run

Large profiles remain sizing-only. Native benchmark integration, full-layer
architecture integration and independently confirmed real-data results are
outstanding.

## Preserved concurrent protocol additions

The latest hot/cold query-frequency skew, guaranteed hot/cold post-interference
probes, and genuinely new-fact targets are retained. All groups save raw counts
and per-sequence outcomes. Repeated query tokens also change learned K/V
streams, so these neural experiments do not hold hidden Key statistics fixed.

Because the hot value is queried much more often, vocabulary-uniform chance
is a misleading baseline. The screen requires both baseline and candidate old
recall to exceed max(1/pairs, observed hot-target fraction) by five percentage
points, and requires both to reach at least 15% new-fact accuracy. These are
conservative diagnostic prerequisites, not proof of key-value binding.
The full requested method-by-seed matrix must finish; missing controls,
unmatched examples, insufficient samples or timeouts remain INCONCLUSIVE.

Here hot means fact index 0; cold groups all other fact indices, with fact 1
forced as a final probe. Cold does not mean never previously queried.
