# Identity + Pointer Doubling — LEAN FORMAL SMALL-SCALE STUDY (2026-11-22)

**ACTIVE:** 340M decoder, 20B total train tokens/model, 4K context, seed 42; only 3 matched baselines + our model, 4 FULL pretrainings. Additional training ablation: only 1 × 2B tokens. No 1.3B experiment; no 3-seed repeat. This document specifies the experiment but the old 3h pilot cannot run it.

## Main 4 full 20B-token trainings (3 baselines + 1 ours)
1. **Transformer:** native tuned dense FlashAttention Transformer, exact total token budget and comparable parameter/FLOP count.
2. **Original Dense Pointer:** coauthor original Identity + Pointer Doubling + no-backtrack/lifting (not Wuke novelty).
3. **Block Sparse Pointer:** original Pointer with completed-block causal sparse routing and exact QK/no-backtrack refinement.
4. **Identity-Hash Sparse Pointer (ours):** causal identity-aware hash routing, full original Pointer learning; Multi-Level Jump Utility evaluated at inference without retraining.

Data: freeze shared 80% real FineWeb-Edu + 20% train-only graph/chain tokens (20B total per arm). Document tokenization, templates, token distribution, matched initialization where compatible, optimizer schedule, model params and FLOPs. Use independent held-out graph families and identifiers, not train/test templated copies.

## ONLY ONE short training ablation: 2B tokens
- **hash_without_identity_conditioning:** train to 2B with the SAME 20B-model learning-rate schedule and token order, compare its graph/retrieval/quality results ONLY to the main Identity-Hash model checkpoint stored at 2B. Do not compare directly to the final 20B checkpoint and call it an effect of hashing. This measures the early-training identity dependence, NOT full-20B ablation validity.

## Sparse and online cache are TWO separate factors (cache policies add ZERO LM trainings)
- **Sparse:** compare full-trained dense, block-sparse and hash-sparse at equal accuracy and real full-model GPU memory/latency; a Dense-TopK score-then-select diagnostic uses the trained dense model without extra training.
- **Cache:** on the SAME frozen 20B-trained neural checkpoint, compare No-Cache, LRU, LFU and Multi-Level Jump-State Utility. Five independent workload traces per policy, cache budgets 12.5/25/50% in matched ACTUAL resident bytes INCLUDING metadata, and dynamic edit rates 0/1/10% with exact cache invalidation.
- **Minimum crossed evaluation:** dense vs identity-hash × no-cache/LFU/utility at matched inference memory and answers; retain LRU as a comparator in the cache-only table. No additional full training for combinations. The existing symbolic JumpCache is not a substitute for a real cross-query neural cache.

## Formal evaluation, unchanged sample rigor
- **Primary:** hold out 2,000 independent graph/chain examples per prespecified major difficulty × order/hop condition (1/2/4/8/16/32 hops; reverse/forward/random/mixed; unique/duplicate keys). Include cycles, aliasing and distractors; explicitly report failures under non-reverse order.
- Standard RULER task-length cells (native 500 each where supported), with 2,000 additional matched independent multi-hop/tracing instances; BABILong native 1,000-question sets when available, GraphQA and held-out real next-token NLL over at least 20M independent tokens.
- GPU kernel and full-model routing/gather/index build costs, prefill/decode wall-time (median/P95), actual peak VRAM allocated/reserved, throughput and accuracy at 2K/4K/8K/16K/32K (64K if supported). Record OOMs.

## Validity constraints and implementation status
- Train exactly one FULL run per arm seed=42; keep identical data order and fixed tokenizer, suitable equivalent architecture/model/optimizer budgets, and use validation only for hyperparameters. Save checkpoint at consumed tokens=2B and 20B for all full 20B runs.
- No claim of seed robustness from test-set bootstrap. Test-set bootstrap with matched held-out examples estimates test sampling variation only.
- Pointer doubling, no-backtrack, identity matching and LSH are established prior ideas; proposed novelty is the SPECIFIC integration and measured compute-reuse benefit, NOT generic hashing or binary lifting. AttentionIndex (arxiv 2611.02508) and ARM (ICML 2026) are related but not direct fair model baselines without matching architectures/training.
- Current reference pointer PyTorch operators still leave ordinary Transformer attention dense. Real sparse GPU kernel, 340M trainer, 20B streaming corpus and cross-request neural Jump Cache are NOT implemented. Do not treat old pilot results as formal findings.

## GPU resources (unmeasured planning)
4 × 340M × 20B plus 1 × 340M × 2B = **82B tokens total read** over the separate training jobs. At a hypothetical sustained H100 200 TFLOP/s and ideal 6PT, ~232.3 H100 core-train hours before routing/index costs, evaluation, reduced utilization or failed runs. True time is unknown until native kernels run. Target >=80GB VRAM, >=128GB host RAM, >=1TB fast workspace. No 1.3B expansion.
