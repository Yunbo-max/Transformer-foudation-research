# Dynamic Shared LM Output Vectors: LOCKED LEAN FULL-TRAINING STUDY (2026-11-22)

**ACTIVE:** 125M parameters, 5B full pretraining tokens/arm, context 2048, ONE training seed=42, precisely THREE competitive or mechanistically essential baselines plus ONE proposed method. Main runs are complete formal training, NOT a 3h pilot. All 1.3B and second-scale expansion deferred.

## Main full training matrix — EXACTLY 4 trained models
1. tuned_output_adam_epsilon — **BASELINE**: strong cheap optimizer-level comparator
2. original_dynamic_weight_kmeans — **BASELINE**: Vuk Rosic existing function-preserving additive shared LM output head
3. syntactic_smoothing — **BASELINE**: EMNLP 2024 supervised syntax-neighbour signal sharing comparator
4. positive_target_gradient_direction_magnitude — **OURS**: historical positive-target gradient-based dynamic groups with direction plus calibrated magnitude

All arms are train-from-scratch on an identical frozen FineWeb-Edu with full GPT-2 50,257-token vocabulary (NO artificial UNK) training stream with matched architecture/parameter budget and fully documented FLOPs. Tune on validation only with equivalent search budget; all four main models consume the FULL 5B tokens. Report source-faithful official protocols and disclose non-identical architecture components. No 4th or 5th baseline implied.

## Minimal short-budget training ablations — EXACTLY 2 additional jobs
- gradient_direction_only: 1B TRAIN tokens, compare with `positive_target_gradient_direction_magnitude` checkpoint at the same consumed token count; does direction add the gain without gradient magnitude?.
- optimizer_reset_control: 1B TRAIN tokens, compare with `positive_target_gradient_direction_magnitude` checkpoint at the same consumed token count; rule out Adam-moment reset / regroup frequency artifact.

At the designated 1B/2B point, SAVE checkpoints of applicable FULL main arms automatically. Additional ablation training follows the SAME training schedule up to that exact token count; no replacement with shorter cosine. Matched early-step results ONLY: **do not compare an ablation trained for 2B with a 20B final main model and claim the same budget**. The short-budget table explicitly states its shorter-data limitation.

## Evaluation (no reduction in held-out sample rigor)
- PRIMARY: independent FineWeb-Edu held-out next-token NLL >=50M tokens excluding no real GPT-2 vocabulary rows; preserve all tokens and report frequency-bin conditional NLL and common-token regression
- At least two held-out out-of-domain text corpora and a genuine low-frequency-token stratified challenge where actual eligible token counts and context coverage are disclosed
- Official BLiMP 67 tasks (typically 1,000 pairs each), plus LAMBADA/ARC and relevant frequency bias/anisotropy
- Monitor grouping costs, assigned frequencies/cluster members, optimizer state effects, folded-out inference throughput, train throughput/VRAM

## Closest-work and interpretation constraints
- Original dynamic weight kmeans/additive output row is Vuk's work: our hypothesis is gradient-based grouping, not the additive sharing itself.
- Syntactic Smoothing EMNLP 2024: https://aclanthology.org/2024.emnlp-main.344/ ; if supervision/architecture cannot be matched, label as distinct recipe and do not assert exact parameter/compute fairness.
- Extra vocab size/tied head/batch/K sweeps deferred; no 350M large-scale run now.

## Fixed scientific controls
- One full training seed 42 per MAIN arm. Within-project arms use matched tokenizer, source sharded train manifest, data order, training token budget, model/optimizer schedule and comparable parameter + training FLOP budget. Run exactly ALL declared main arms; do not omit weak ones.
- For the limited short training ablations, record and preserve MAIN model checkpoints **after the same 1B/2B number of consumed tokens**. The short ablation uses the identical full-run LR schedule through that token count (not a separately accelerated cosine schedule). Compare with the matched-token MAIN checkpoints, NEVER with MAIN final 5B/20B checkpoints. Short-budget ablations only show early-training mechanism sensitivity; they do not establish the full-budget mechanism in isolation.
- Validate methods/official kernel and numerical invariants before burning pretraining compute. This is engineering correctness verification, NOT a reduced-data scientific pilot. No numerical training result is currently claimed.
- For full checkpoints, run official native benchmarks with unmodified evaluation metrics and task-specific sample counts. Use the same held-out prompts/examples across arms; preregister primary metrics. Report test-set paired/bootstrap confidence intervals where appropriate, and explicitly state ONE training seed cannot establish training-seed robustness. Avoid data/test leakage.
- Measure effective training token throughput, full-model and operator kernel cost, wall time, actual GPU allocated/reserved peak, host RAM, and test accuracy/NLL. Failed/OOM cells stay in results; efficiency claims must be from real GPU, not proxy FLOP counts.
- IMPORTANT: This file and SMALL_SCALE_STUDY.json are experiment-design artifacts. The legacy 3h scripts do NOT train 125M/340M on 5B/20B tokens, and the needed native training, kernels, datasets and evaluation integration are not yet implemented. No ready-to-run full training is asserted.

## Scope and resources (engineering estimate only)
- FULL runs: 4 × 5B tokens = 20B consumed tokens. Short ablations: 1B + 1B = 2B. TOTAL 22B token-read volume (may repeat the same data shards across different methods).
- Theoretical core training H100 GPU hours if (only IF) 6PT FLOPs and sustained 200 TFLOP/s: 22.9h. This IS NOT measured training time, excludes evaluation and any extra routing/matrix compute, and does NOT cap runtime for currently unfused operators.
- Target VRAM 80GB for 125M methods with custom kernels (output head may use 24–48GB), host RAM 64–128GB, streaming data. Main code requires work before this protocol can execute.
