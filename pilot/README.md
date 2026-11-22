> **Paper-level experiments (one full training run per configuration, comparable published training budgets):** see [FULL_PAPER_EXPERIMENTS.md](FULL_PAPER_EXPERIMENTS.md).
> This is a complete research protocol, **not yet a runnable paper-scale training pipeline**. Existing short-run scripts do not implement it.

# NEW Project 11 acceleration extensions (2026-11-22)

Based on the original coauthor 04 Identity/Pointer and 05 Pointer Doubling,
this branch now contains TWO independent acceleration hypotheses plus their
crossed ablation. Existing original-method pilot below is unchanged.

- **Sparse Pointer Attention**: causal block routing AND identity-aware
  content-hash routing, with exact QK/no-backtrack refinement. See
  pointer_accel.py and pointer_hash.py.
- **Online Jump-State Cache**: resident budget over (node,2^level)-hop
  composed pointers, with LRU/LFU/random/level_LRU and Multi-Level Jump-State
  Utility admission. See jump_cache.py.
- **3x2 crossed experiment**: Dense/Block/Hash × No Cache/Utility Cache on
  identical frozen-embedding multi-hop traces. See run_accel.py.
- **Optional original paper integration**: integrate_pointer.py and
  run_sparse_model.py compare original pdg.py vs block/hash sparse P/Q/R internal
  updates with identical training, both seeds. Other Transformer heads
  remain DENSE, so no model-wide speed/VRAM claim.
- **All equations, caveats, benchmark rules**: [ACCEL_METHOD.md](ACCEL_METHOD.md).

Local numerical/reference tests were checked; GitHub GPU training and
hardware profiling have NOT been executed. Synthetic proxy evidence alone
is never a new LLM result.

    python3 -m unittest discover -s pilot -p 'test_*.py'
    python3 pilot/run_accel.py --suite all --device cpu
    # Full strict <=3h launcher for a Linux NVIDIA machine:
    bash pilot/run_accel_3h.sh cuda

Outputs: .pilot_workspace_pointer_accel/reports/accel.json and
.pilot_workspace_pointer_sparse_model/reports/sparse_model_decision.json.

---

# Project 11 next-stage novelty gate: [GATE11.md](GATE11.md)

The original reverse-order doubling/no-backtrack/lifting is prior coauthor work. The remaining Wuke research gap is order-agnostic, variable-hop composition, which is **not implemented** here; existing pilot is a stress test only.

# Pilot D — Identity + Pointer Doubling as ONE Paper (<= 3 hours)

Pinned MIT source:
https://github.com/vukrosic/llm-research-paper-drafts/tree/main/05-attention-pointer-doubling
SHA d0dc052749a97061f6daa6907371bd0a2ec33405

The combined method comprises identity matching (prior Memory Mosaics),
independent pointer propagation, pointer-doubling (A @ P rather than A @ E),
doubled no-backtrack, and gated binary lifting. The hypothesis being screened
is whether this mechanism can extend beyond reverse-ordered synthetic chains.

## Run

Preferred strict three-hour GNU/Linux command (includes checkout and data prep):
  bash pilot/run_3h.sh --device cuda  # omit --device for Branch reference

  pip install torch numpy
  python3 pilot/run.py --device cpu --budget-seconds 10500
  python3 pilot/run.py --device cuda --budget-seconds 10500

No data download required: the pinned source synthesizes final-answer-only
multi-hop recall. Two paired seeds (0,1), model width64, 3 layers, 4 heads,
8 pairs, 75 training steps; source has per-run eval and fixed 256 eval rows.
Input order is NEVER silently changed between paired variants.

## Primary cells
- reverse7: plain, one-hop+no-backtrack, doubling+lifting without no-backtrack,
  full lifting+no-backtrack; this tests the claimed 3-layer/7-link advantage.
- forward4: plain, full lifting, predecessor doubling; tests causal order.
- random7: plain, one-hop, full lifting; tests order generalization.
- reverse5: lifting alone vs full; tests non-power-of-two jumps.
- repeat7: one-hop vs full with filler tokens and 30% duplicated chain words.

Primary metric is exact final answer accuracy; secondary metrics are
training steps to 90% accuracy, time and dropout on order changes.
Advance to realistic chain benchmarks ONLY if, in BOTH seeds, reverse7 full
has >=90% and beats both plain and one-hop by >=40 percentage points, and
random7 full and forward4 predecessor both achieve >=80%. If only reverse
passes, label MECHANISM_ONLY_NO_GENERALIZATION (not an LM improvement).
No success -> redesign or stop.

Source code is cloned/pinned into a local ignored work directory; do not
mistake the method's short synthetic run for natural-language reasoning.
No full LM pretraining was performed. Work directory outputs:
.pilot_workspace_pointer/reports/{report.json,decision.json,case stdout/stderr}.
Runner hard budget 10500 s, excluding cloning and package setup.
