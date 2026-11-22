# SG-Delta: LOCKED LEAN FULL-TRAINING STUDY (2026-11-22)

**ACTIVE:** 340M parameters, 20B full pretraining tokens/arm, context 4096, ONE training seed=42, precisely THREE competitive or mechanistically essential baselines plus ONE proposed method. Main runs are complete formal training, NOT a 3h pilot. All 1.3B and second-scale expansion deferred.

## Main full training matrix — EXACTLY 4 trained models
1. gdn2 — **BASELINE**: native strong decoupled erase/write
2. qed — **BASELINE**: 2026 query-derived erase direction; closest query-informed update
3. kda — **BASELINE**: channel-wise decay/erase-write tied predecessor, source-faithful kernel
4. sg_delta_query_joint — **OURS**: historical query geometry with directional decay + protected writing

All arms are train-from-scratch on an identical frozen FineWeb-Edu training stream with matched architecture/parameter budget and fully documented FLOPs. Tune on validation only with equivalent search budget; all four main models consume the FULL 20B tokens. Report source-faithful official protocols and disclose non-identical architecture components. No 4th or 5th baseline implied.

## Minimal short-budget training ablations — EXACTLY 2 additional jobs
- sg_query_write_only: 2B TRAIN tokens, compare with `sg_delta_query_joint` checkpoint at the same consumed token count; remove query-derived directional decay.
- sg_query_decay_only: 2B TRAIN tokens, compare with `sg_delta_query_joint` checkpoint at the same consumed token count; remove query-guided write protection.

At the designated 1B/2B point, SAVE checkpoints of applicable FULL main arms automatically. Additional ablation training follows the SAME training schedule up to that exact token count; no replacement with shorter cosine. Matched early-step results ONLY: **do not compare an ablation trained for 2B with a 20B final main model and claim the same budget**. The short-budget table explicitly states its shorter-data limitation.

## Evaluation (no reduction in held-out sample rigor)
- PRIMARY: independently held-out old/new fact retention-plasticity with >=2,000 task instances per preregistered primary condition (old-after-interference, new-key/same-value alias, same-key replacement, hot/cold write conflict)
- Natural held-out FineWeb-Edu NLL >=20M tokens plus WikiText-103, LAMBADA, ARC, HellaSwag, PIQA, BoolQ using fixed protocols
- Official RULER at 4K/8K/16K/32K where supported; 500 per native task/length and separate >=2,000 priority multi-key/variable-tracing tests
- Measure state statistic C (KxK), Cholesky, kernel time, real training/decode latency, throughput and GPU memory

## Closest-work and interpretation constraints
- Source-near reference GDN-2: https://github.com/NVlabs/GatedDeltaNet-2
- QED: https://arxiv.org/abs/2608.13668 (code availability / faithful reimplementation must be confirmed)
- KDA: published strong state-memory predecessor; no claim it is the most recent 2026 method
- Muir (arxiv 2611.08376) is newer and highly relevant but combines a different low-rank gate/readout/FFN system; list as external related SOTA, NOT a silently dropped comparator and NOT a fourth mandated training. Reconsider only with a verified reproducible matched GPU implementation.
- SG full-matrix query geometry currently lacks efficient non-diagonal NVIDIA kernel; full-train wall time is UNKNOWN without integration and profiling.

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
