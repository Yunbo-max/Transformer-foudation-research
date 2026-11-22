# Identity + Pointer Doubling — ACTIVE SMALL-SCALE FULL-PRETRAINING STUDY (2026-11-22)

**SCOPE LOCK:** 340M decoder, 20B training tokens, 4096-token train context, **ONE full train/architecture (seed 42)**. No 1.3B/40B large-scale training now. Proposed 80% natural FineWeb-Edu + 20% generated multi-hop/graph training (train templates/keys disjoint from all evaluation graphs). Current short pdg.py and symbolic JumpCache are NOT runnable whole-model systems.

## Closest standards and novelty boundaries
Sanford/Hsu/Telgarsky ICML 2024 and other graph-theory works already cover logarithmic-depth pointer composition; original coauthor 04+05 already implements identity matching, pointer doubling, no-backtrack and lifting. Wuke hypotheses are efficient causally sparse Pointer routing and **real** online multi-level computed jump-state caching; hash/LSH itself is prior art. Do not claim generic causal pointer doubling is order-agnostic.

## EXACT full-training matrix — 4 jobs
All 340M / 20B / 4K / one seed 42, with model params, actual FLOPs and train data mixture reported:
1. Standard tuned dense Transformer
2. Original coauthor Dense Identity + Pointer Doubling
3. Coauthor Pointer + causal block-routed Sparse Pointer
4. Coauthor Pointer + identity-aware hash-routed Sparse Pointer
Same no-backtrack and doubling controls except where operator sparse candidates differ; preserve causality. Dense-topK as non-trained inference diagnostic (computes dense scores first, never billed sparse). New order-agnostic encoder/passes are not implemented and outside this four-job scope.

## Full formal testing of all trained checkpoints (not proxy)
- **Primary:** final-answer exact accuracy on independently held-out graph families. Number of nodes 16/64/256; hops 1/2/4/8/16/32; reverse/forward/random/mixed order; unique vs duplicated keys; distractors and cycles. >=2,000 independent held-out examples per PRIORITIZED order/hop/difficulty cell, with disjoint key inventories and graph instances. Report failed forward/random order rather than omit it.
- **External:** official RULER tracing/aggregation/multi-hop across 4K/8K/16K/32K (500 native samples/task/length where supported), independent 2,000 additional examples for priority tracing; native BABILong 1,000-task split by 2K-32K where benchmark provides it; held-out GraphQA; natural held-out NLL >=20M tokens.
- **Sparse factor:** compare dense vs block vs hash on trained models, and dense-topK inference diagnostic. Measure candidate recall, exact answers, routing/index/gather FLOPs/costs, actual whole-model wall latency, training/inference peak GPU memory, not only pointer score-pair counts.
- **Cache factor (EVALUATION ONLY, no extra training):** on same stored checkpoint and same prefix, implement genuinely reusable multi-query neural jump states; compare none/LRU/LFU/random/level-LRU/Multi-Level Utility at matched **physical bytes** (12.5/25/50% of a declared full-state reference). FIVE independent request traces per policy with 0/1/10% edits, Q=1/8/64/512. Verify correctness after edits and include metadata, evictions, prefill and decode in memory/time.
- **Crossed factor:** dense/block/hash × cache none/utility, plus strongest baseline LRU and LFU in every sparse arm. Record within-trace paired accuracy/time/cost, model-level speedups; context 2K/4K/8K/16K/32K/64K and batch 1/4/16 as feasible (explicit OOM and regression reports). 20 warmups +100 timed repetitions per supported hardware configuration. Multiquery prefix reuse must not reuse stale states after corpus/prefix change.
- **Training scope:** 4 trainings only. Cache policies do NOT train separate LM checkpoints. Any new order-agnostic architecture, added full-training cache policy, or 1.3B replication is deferred. Existing original attention and residual paths are still dense until truly replaced.

## Non-negotiable execution and evaluation rules
- This is **complete small-scale pretraining**, not 50/400-step pilot: actually consume the declared token budget and save loss curves/checkpoints. One training seed (42) for each arm. Pair dataset shards, data order, tokenizer, initialization of comparable tensor shapes, token budget, architecture width/parameter count, optimizer/schedule and compute reporting. Never fabricate between-training-seed variance.
- Reserve validation for equal-budget hyperparameter selection BEFORE fixed final tests; define and freeze primary metric, held-out corpora, prompts and success criteria. Report all baselines, even worse ones. Held-out item bootstrap or document-cluster bootstrap concerns TEST sample uncertainty only; it does not establish training-seed robustness.
- Each official benchmark must use the official prompt/evaluation code and labeled native sample counts; custom supplements are reported in separate columns. Disjoint tokenizer train/test corpora; for synthetic graph tasks disjoint instances, held-out key inventories and graph structures. Avoid contamination.
- Report full-training wall time, effective tokens/sec, FLOPs inclusive of extra operators, model parameters, peak GPU **allocated/reserved** VRAM, host RAM, checkpoint footprints, prefill/decode median/P95 and test accuracy/NLL. All data pipelines must be sharded and reproducible.
- All rows below are **PLANNED**. Existing pilot scripts **DO NOT** implement the 125M/340M 5B/20B full-scale training pipelines. No complete GPU result is claimed. This stage has one model size only; larger scales and any extra training-required sensitivities are **DEFERRED**, not silently included in the time total.

## H100 80GB resource estimate (not benchmarked)
340M/20B at an optimistic assumed 200TFLOP/s and 6*P*T gives 56.7 ideal GPUh/job, 4 jobs = **~227 ideal H100 GPUh** core. With unfused sparse indexing, dense remaining heads, long-context inference and cache evaluation, reserve **~480–2,100 H100 GPUh** conditional on an optimized GPU sparse kernel and truly neural cache implementation; otherwise no reliable upper bound. GPU >=80GB and host RAM >=128GB, scratch >=1TB and stream datasets. No large-scale second model included.
