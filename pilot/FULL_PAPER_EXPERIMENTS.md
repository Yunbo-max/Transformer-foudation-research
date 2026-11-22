# Predicted-Skip: LOCKED LEAN FULL-TRAINING STUDY (2026-11-22)

**ACTIVE:** 340M parameters, 20B full pretraining tokens/arm, context 4096, ONE training seed=42, precisely THREE competitive or mechanistically essential baselines plus ONE proposed method. Main runs are complete formal training, NOT a 3h pilot. All 1.3B and second-scale expansion deferred.

## Main full training matrix — EXACTLY 4 trained models
1. official_gdn2 — **BASELINE**: native unmodified GDN-2, essential direct comparator
2. qed — **BASELINE**: 2026 query-informed interference-erasure SOTA comparator
3. gdn2_previous_query_write_only — **BASELINE**: closest skip mechanism with value write suppression but no erase scaling
4. gdn2_previous_query_joint — **OURS**: GDN-2-port of joint erase/write answer feedback, with native guard parity

All arms are train-from-scratch on an identical frozen FineWeb-Edu training stream with matched architecture/parameter budget and fully documented FLOPs. Tune on validation only with equivalent search budget; all four main models consume the FULL 20B tokens. Report source-faithful official protocols and disclose non-identical architecture components. No 4th or 5th baseline implied.

## Minimal short-budget training ablations — EXACTLY 2 additional jobs
- previous_query_shuffled_control: 2B TRAIN tokens, compare with `gdn2_previous_query_joint` checkpoint at the same consumed token count; does previous-query information matter beyond extra parameters?.
- current_key_read_control: 2B TRAIN tokens, compare with `gdn2_previous_query_joint` checkpoint at the same consumed token count; replace previous-query read by current-key feedback.

At the designated 1B/2B point, SAVE checkpoints of applicable FULL main arms automatically. Additional ablation training follows the SAME training schedule up to that exact token count; no replacement with shorter cosine. Matched early-step results ONLY: **do not compare an ablation trained for 2B with a 20B final main model and claim the same budget**. The short-budget table explicitly states its shorter-data limitation.

## Evaluation (no reduction in held-out sample rigor)
- PRIMARY: old knowledge retention AND new fact plasticity after answer re-entry, 2,000 held-out cases each for overwrite/alias/hot-cold/changed answer; false skip rate and event-conditioned suppression
- Native FineWeb-Edu NLL >=20M tokens, WikiText-103, LAMBADA, commonsense QA and retrieval tasks with fixed native prompt/eval
- Native RULER 500 samples per supported task/length 4K-32K plus 2,000 independent priority NIAH/multi-key cases
- Official NVIDIA GDN-2 forward/backward/guard parity and end-to-end GPU token throughput, prefill/decode latency, allocations, extra kernel work

## Closest-work and interpretation constraints
- Vuk Rosic's original Predicted-Skip already includes write and joint erase/write suppression (lpsE=1). The joint mechanism itself is PRIOR ART. A GDN-2 port alone is not established paper-level originality.
- QED native official release or faithful independent reimplementation must be confirmed. Do not conflate mismatched update rules.
- Soft skip is not actual reduced FLOPs unless a concrete kernel avoids work.
- Initial official kernel parity validation must pass before a full run.

## Fixed scientific controls
- One full training seed 42 per MAIN arm. Within-project arms use matched tokenizer, source sharded train manifest, data order, training token budget, model/optimizer schedule and comparable parameter + training FLOP budget. Run exactly ALL declared main arms; do not omit weak ones.
- For the limited short training ablations, record and preserve MAIN model checkpoints **after the same 1B/2B number of consumed tokens**. The short ablation uses the identical full-run LR schedule through that token count (not a separately accelerated cosine schedule). Compare with the matched-token MAIN checkpoints, NEVER with MAIN final 5B/20B checkpoints. Short-budget ablations only show early-training mechanism sensitivity; they do not establish the full-budget mechanism in isolation.
- Validate methods/official kernel and numerical invariants before burning pretraining compute. This is engineering correctness verification, NOT a reduced-data scientific pilot. No numerical training result is currently claimed.
- For full checkpoints, run official native benchmarks with unmodified evaluation metrics and task-specific sample counts. Use the same held-out prompts/examples across arms; preregister primary metrics. Report test-set paired/bootstrap confidence intervals where appropriate, and explicitly state ONE training seed cannot establish training-seed robustness. Avoid data/test leakage.
- Measure effective training token throughput, full-model and operator kernel cost, wall time, actual GPU allocated/reserved peak, host RAM, and test accuracy/NLL. Failed/OOM cells stay in results; efficiency claims must be from real GPU, not proxy FLOP counts.
- IMPORTANT: This file and SMALL_SCALE_STUDY.json are experiment-design artifacts. The legacy 3h scripts do NOT train 125M/340M on 5B/20B tokens, and the needed native training, kernels, datasets and evaluation integration are not yet implemented. No ready-to-run full training is asserted.

## Scope and resources (engineering estimate only)
- FULL runs: 4 × 20B tokens = 80B consumed tokens. Short ablations: 2B + 2B = 4B. TOTAL 84B token-read volume (may repeat the same data shards across different methods).
- Theoretical core training H100 GPU hours if (only IF) 6PT FLOPs and sustained 200 TFLOP/s: 238.0h. This IS NOT measured training time, excludes evaluation and any extra routing/matrix compute, and does NOT cap runtime for currently unfused operators.
- Target VRAM 80GB for 340M methods with custom kernels (output head may use 24–48GB), host RAM 64–128GB, streaming data. Main code requires work before this protocol can execute.
