# Predicted-Skip GDN-2 — Joint Erase + Write Suppression

Status (2026-11-21): PyTorch CPU tests and tiny synthetic training passed locally;
the official NVIDIA CUDA kernel path and paper-scale experiments have NOT
been GPU-verified. This project is INDEPENDENT of SG-Delta.

Credits: prior Predicted-Skip mechanism by coauthor Vuk Rosic:
https://github.com/vukrosic/llm-research-paper-drafts/tree/main/03-linear-attention-skip-lookups
His MIT reference is not copied here. The proposed NEW variant suppresses both
erase and write of NVIDIA GDN-2; do not call the write-only control our invention.

## Mathematical definition

GDN-2 uses S in R^{KxV}, with independent key-side erase b and
value-side write w, and channel-wise log decay g:

    Sbar_t = diag(exp(g_t)) S_{t-1}
    S_t = Sbar_t - k_t [(b_t*k_t)^T Sbar_t] + k_t (w_t*v_t)^T

At the start of each chunk, take ONLY the already computed recurrent state S_c.
For t in that chunk use the previous query (q_{t-1}) and current value v_t:

    r_t = S_c^T q_{t-1}
    rho_t = exp(-gamma * max(0, cosine(r_t,v_t)))

Main proposal:

    b_new,t = rho_t*b_t
    w_new,t = rho_t*w_t
    g_new,t = g_t  (decay unchanged)
    S_t = Sbar_t + rho_t * (S_t^{original active edit}-Sbar_t)

This identity is exact for the recurrence WITHOUT extra internal gate projection.
The official CUDA implementation has an optional beta_guard safety projection;
we initially DISABLE it in the matched parity study, then MUST report an
additional default-beta_guard-on validation for actual NVIDIA behavior.
Setting rho=0 removes active edit but DOES NOT turn off decay.

## Quick run from repository root

Requires Python 3.10+, torch, and git only for optional official checkout.

    python3 -m unittest discover -s pilot -p 'test_*.py'
    python3 pilot/oracle_diagnostic.py
    python3 pilot/run.py --profile smoke --methods baseline joint --seeds 0
    bash pilot/run_3h.sh --device cuda --backend reference

CPU can be tested with --device cpu. The explicit GNU/Linux launcher enforces
10,800 seconds total (3 hours) and an inner 10,200-second run budget.
Outputs: runs/predskip_joint/manifest.json, per-seed arm JSON, decision.json,
and error logs. Incomplete comparisons remain INCONCLUSIVE. This runner does
not launch remote training or create a background job.

Optional official NVIDIA GDN-2 operator (not automatically installed):

    git clone https://github.com/NVlabs/GatedDeltaNet-2.git
    git -C GatedDeltaNet-2 checkout de8476de0fecae08fc16f0bf3154900244c61c73
    # Install the dependencies according to that pinned NVIDIA README.
    python3 pilot/validate_official.py --official-root ./GatedDeltaNet-2
    bash pilot/run_3h.sh --device cuda --backend official --official-root ./GatedDeltaNet-2

Do not bypass the numeric parity test. The official backend calls unchanged
NVIDIA chunk_gdn2, but the surrounding decoder/projections are a SMALL
reference model, not the full NVIDIA 1.3B architecture. Repeated chunk kernel
calls and fetching chunk-start states may be slower than native full-sequence
GDN-2. Measure tokens/sec, not just accuracy. NVIDIA code is not vendored and
has NVIDIA Source Code License-NC, NOT MIT.

## Model sizing

| Stage | Decoder / heads / key dim | Actual parameters | Status |
| --- | --- | ---: | --- |
| smoke | D32 / L1 / H1 / K8 | 14,713 | locally smoke-tested |
| 3-hour pilot | D128 / L2 / H2 / K16 | 388,802 | locally smoke-tested |
| stress | D192 / L3 / H3 / K32 | prints at runtime | not full-run validated |
| 100M-class sizing | D768 / L12 / H12 / K64 | 107,859,468 | sizing only |
| 340M-class sizing | D896 / L24 / H14 / K64 / vocab32K | 346,617,624 | sizing only |

    python3 pilot/run.py --profile 100m-sizing --methods joint --seeds 0 --dry-run
    python3 pilot/run.py --profile 340m-sizing --methods joint --seeds 0 --dry-run

Large modes are BLOCKED pending CUDA parity and efficient full-layer
integration. Pilot recommendation: 1x RTX 4090 24GB and host RAM 16-32GB.
Later 100M: 1x 80/97GB H100-class GPU if verified; 346M paper-class requires
hardware/throughput profiling, potentially multi-GPU.

## Data protocol and scoring

Pilot: online causal synthetic MQAR, 8 initial Key-Value facts, 8 repeated
queries (deliberately hot/cold skewed), 4 later interfering facts, plus
causal answer queries for TWO genuinely NEW post-interference facts, and
NEW ALIAS KEY whose Value is the
same as an earlier answer. Data generated on-the-fly with fixed seeds; no
model is passed the answer-position labels. Loss is teacher-forced causal
next-token cross entropy only at marked query-answer targets.
Evaluation uses separate seeds and panels: 0/4 distractor facts and
twice as many original fact pairs. Initial train 150 steps, batch 8, seeds 0,1.

Core ablations (all same neural parameters and input sequences):
- baseline: original GDN-2 recurrence, no skip
- write: coauthor Predicted-Skip scales only w
- joint: NEW main proposal scales both b and w
- shuffled: content-mismatched value comparison, joint control
- keyread: uses S_c^T k_t in place of S_c^T q_{t-1}, timing control

Primary metrics: old fact recall after distractor writes (overall/hot/cold),
NEW-fact acquisition accuracy, new alias-Key recall, answer accuracy/NLL,
and retention-vs-plasticity tradeoff. Mechanism:
mean rho at ANSWER token positions vs FACT value positions, fraction with
rho<.5, gamma learned, throughput, wall time, VRAM. The gate event categories
are used for EVALUATION ONLY, never as predictor inputs.

Warning: a value can already have been retrieved but still belong to a
genuinely NEW Key-Value association. The analytic oracle_diagnostic.py is
an explicit counterexample showing Joint Skip may reject such an update.

Prespecified 3-hour screen heuristic: in both complete seeds, Joint must beat
both Write-only and baseline by >=3pp old-post recall, and new alias-Key
accuracy and genuinely NEW-fact acquisition must each be no worse than 3pp
below baseline. Both post-interference old recall and new-fact accuracy
must exceed 15%, or the result is INCONCLUSIVE_NOT_LEARNED. Do not call passing one
test paper-grade evidence; repeat >=3 seeds and plot precision/recall of skip
decisions. Chance-level models are INCONCLUSIVE.

## Stage 2 planned ONLY if the screen passes

1. GPU forward/state/autograd parity with NVIDIA chunk_gdn2.
2. Compare official GDN-2 baseline with identical kernel dispatch (same
   chunking for ALL arms), then quantify additional chunk-state latency.
3. ~108M model on document-disjoint streamed FineWeb-Edu plus held-out
   answer-copy and new-Key/same-Value probes. Train >=100M tokens as a
   follow-up budget; 3 hours NOT guaranteed.
4. ~346.6M GDN-2 class trained under matched tokens, plus RULER multi-key
   NIAH, JRT repeated-recall, 2K/4K/8K/16K, validation NLL/PPL,
   mean rho by token type, time/tokens and peak VRAM.

Public references:
https://github.com/NVlabs/GatedDeltaNet-2
https://github.com/NVIDIA/RULER
https://huggingface.co/datasets/HuggingFaceFW/fineweb_edu_100BT
No trained official GDN-2 performance claim has been made.
