# Dynamic Shared LM Output Vectors — ACTIVE SMALL-SCALE FULL-PRETRAINING STUDY (2026-11-22)

**SCOPE LOCK:** 125M REAL decoder, 5B pretraining FineWeb-Edu tokens, context 2048, full GPT-2 50,257-vocab (no artificial UNK), **ONE full training per arm / seed 42**. No 350M/20B or 1.3B stage for now. Old 32-width/400-step synthetic study is NOT paper-grade.

## Closest real research
Original additive shared-output dynamic weight KMeans (Vuk Rosic coauthor) is PRIOR WORK; our contribution hypothesis is historical positive-target gradient direction and direction+magnitude grouping. Syntactic Smoothing (EMNLP 2024) is a required, source-faithful competitor. Preserve function-equivalent output rows at every regroup, audit optimizer-state/reset confounding and full-group assignment costs.

## EXACT primary FULL-training matrix — 8 jobs
All matched 125M / 5B tokens / 2048 context / GPT-2 vocab / identical token stream / one seed 42:
1. Tuned AdamW epsilon head-only (unshared)
2. Fixed Random shared output groups
3. Fixed Frequency shared output groups
4. Dynamic learned Weight K-Means (original coauthor)
5. Full-gradient EMA K-Means
6. Target-conditioned Positive-Gradient Direction grouping
7. Positive-Gradient Direction + Log-Magnitude grouping
8. Syntactic/Morphological Smoothing source-faithful competitor (disclose any extra POS supervision)
Same decoder stack, tied/untied design within comparators, optimizer schedule, matched token budget, and fair validation-only tuning. Stop only after full 5B tokens, not an early 400-step screen.

## Complete evaluation for EACH checkpoint
- **Primary:** NLL on >=50M independent held-out FineWeb-Edu tokens; additionally >=2 out-of-domain corpora. Full-tokenizer NLL, frequency-conditional token loss and type-balanced token analysis, in-vocab rare cohorts and frequent-token regressions. Do NOT invent 1–9-occurrence cohorts if a 5B-token training corpus gives no suitable tokens.
- **Rare generalization:** preregister a separate held-out rare-token context suite with a TARGET of >=2,000 distinct eligible token types and >=10 held-out contexts per type; record actual coverage and sampling source. Report population weighted and balanced-type scores separately.
- **Linguistic/general:** full BLiMP (67 subtasks x official 1,000 pairs, no handcrafted replacement), HellaSwag, ARC and LAMBADA (supplementary). Full logits and PPL after folding the shared head.
- **Mechanism & efficiency:** group-size/semantic coherence, historical gradient direction vs full gradient vs learned weights; optimizer-reset control and extra parameter accounting. Match inference tokens/sec BEFORE vs AFTER folding shared vectors, training throughput and peak GPU VRAM. Additional training to sweep K, regroup interval, tied/untied, batch size and 100K vocabulary are DEFERRED from this stage rather than silently counted in 8 trainings.

## Non-negotiable execution and evaluation rules
- This is **complete small-scale pretraining**, not 50/400-step pilot: actually consume the declared token budget and save loss curves/checkpoints. One training seed (42) for each arm. Pair dataset shards, data order, tokenizer, initialization of comparable tensor shapes, token budget, architecture width/parameter count, optimizer/schedule and compute reporting. Never fabricate between-training-seed variance.
- Reserve validation for equal-budget hyperparameter selection BEFORE fixed final tests; define and freeze primary metric, held-out corpora, prompts and success criteria. Report all baselines, even worse ones. Held-out item bootstrap or document-cluster bootstrap concerns TEST sample uncertainty only; it does not establish training-seed robustness.
- Each official benchmark must use the official prompt/evaluation code and labeled native sample counts; custom supplements are reported in separate columns. Disjoint tokenizer train/test corpora; for synthetic graph tasks disjoint instances, held-out key inventories and graph structures. Avoid contamination.
- Report full-training wall time, effective tokens/sec, FLOPs inclusive of extra operators, model parameters, peak GPU **allocated/reserved** VRAM, host RAM, checkpoint footprints, prefill/decode median/P95 and test accuracy/NLL. All data pipelines must be sharded and reproducible.
- All rows below are **PLANNED**. Existing pilot scripts **DO NOT** implement the 125M/340M 5B/20B full-scale training pipelines. No complete GPU result is claimed. This stage has one model size only; larger scales and any extra training-required sensitivities are **DEFERRED**, not silently included in the time total.

## H100 80GB budget (estimated, not benchmarked)
125M/5B ~5.2 theoretical H100 GPUh/run at an assumed 200TFLOP/s, eight full jobs = **~42 ideal H100 GPUh** core. A small model underutilizes H100 and dynamic KMeans/grouping adds overhead: reserve **~125–630 H100 GPUh** for full training + large evaluation; this is not measured. 24GB RTX 4090 can likely host training with gradient accumulation; recommend 48GB for comfort, RAM >=64GB and >=500GB scratch. No 350M training is counted.
