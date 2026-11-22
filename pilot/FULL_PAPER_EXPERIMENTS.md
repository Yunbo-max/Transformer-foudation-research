# SG-Delta — FULL PAPER EXPERIMENT PROTOCOL (2026-11-22)

STATUS: COMMITTED STUDY DESIGN ONLY. Full paper-scale pipeline, non-diagonal efficient kernel, native benchmark integration, and all GPU results remain UNIMPLEMENTED/UNVERIFIED. This file does not convert the existing short diagnostic into a paper-grade experiment.

## Closest papers and scale
- Gated DeltaNet-2, arxiv 2605.22791 / NVlabs/GatedDeltaNet-2: 1.3B, 100B FineWeb-Edu tokens, 4K training context, global batch ~0.5M tokens; native RULER and retrieval.
- A Systematic Analysis of Hybrid Linear Attention, arxiv 2507.06457: 340M/20B and 1.3B/100B FineWeb-Edu training.
- The Query Knows What to Forget (QED), arxiv 2608.13668: 340M/15B, five arms, two seeds; direct query-derived erase competitor.

## Full training matrix (3 seeds per training cell)
- 340M parameter matched-scale decoder, 20B actual training tokens, 4K context, frozen tokenizer/training shards. Nine arms x 3 seeds = 27 FULL trainings: official GDN-2, KDA, QED, isotropic Delta reference, query-write, query-decay, query-joint (proposed), key-joint, channel-decay. Same layer count, model width, budget, short-convolution policy and real baseline-specific kernels.
- 1.3B parameter matched-scale decoder, 100B training tokens, 4K context, three arms x 3 seeds = 9 FULL trainings: official GDN-2, strongest QED configuration, query-joint. Hybrid vs purely recurrent stack must be treated as separate experimental configurations, never mixed within a claim.
- Train from scratch under identical FineWeb-Edu token budget. Reference optimizer: AdamW; use official matched recipe (peak LR 4e-4, wd 0.1, global batch about 0.5M tokens, 1B-token warm-up, cosine), with an equal and predeclared validation-only LR search for new/competing methods. Different valid official settings are carefully reproduced in a separate source-faithful table.

## Mandatory evaluations, fixed for all 3 model seeds
1. Native zero-shot language quality: Wikitext-103 PPL, LAMBADA accuracy/PPL, ARC-Easy/Challenge, HellaSwag, PIQA, OpenBookQA, BoolQ; natural held-out FineWeb-Edu NLL on >=20M independent tokens.
2. RULER official suite: 13 tasks x 4K/8K/16K/32K (and 64K where supported), 500 independently generated tests per task-length-model checkpoint to match common official evaluation; matched held-out prompts. Independently test primary S-NIAH-2/3, MK-NIAH and variable tracing with >=2,000 held-out examples each at primary context lengths, no selection after looking at results.
3. Old/new associative recall: test latest-value update, same-key overwrite, new-key/same-value alias, hot/cold access, long interference. >=2,000 independent sequences per PREDECLARED primary condition for each seed, different key/value inventories and no labels in inputs. Verify joint retention vs plasticity and tradeoffs. Include MQAR and document recall (FDA, SQuAD, TriviaQA, NQ, DROP) with native prompts.
4. Ablations: query-write vs query-decay vs query-joint; key geometry vs query geometry; covariance EMA timescale, statistics stop-grad, decay stability, state dimensionality K, context 4K-32K. Do not assume projected-query count is semantic utility.
5. Hardware: profile train throughput tokens/s, peak allocated/reserved VRAM, recurrent+statistic KxK memory, prefill/decode throughput and convergence per training token; compare FULL native implementation incl KxK and Cholesky overhead, vs official GDN-2 and QED.

## Non-negotiable rules (full-paper, not feasibility screening)
- Minimum THREE independently initialized TRAINING runs for EVERY main result, paired seeds {11,22,33} across arms with identical frozen data/order, tokenizer, optimizer family, train tokens and parameter budget. Report all three values, mean, standard deviation; preserve seeds and checkpoints. A bootstrap over questions NEVER substitutes for training seeds.
- Pre-register a single primary metric and contrasts. Freeze train/val/test disjointness and test prompts before running. Validation selects hyperparameters (per-arm equal tuning budget), never test. Report natural-distribution scores AND hard conditional groups; no post-hoc selective tables.
- Synthetic benchmark generators must use independently held-out graph instances/keys and fixed generation seeds common across methods; report sample counts in every table. Two-sided paired bootstrap on test instances (cluster by document/graph), seed-level variability separately, and appropriate multiplicity correction for many tests.
- Log wall clock, throughput, actual peak GPU allocated/reserved VRAM, FLOPs including routing/index construction, host RAM, model size, accuracy/NLL and checkpoints. Run on real NVIDIA GPUs. No CPU/symbolic cached-byte or pair-count proxy can be reported as measured GPU speedup.
- Use full training, never tiny 50-step or 400-step runs for claims. All stated experiments below are PLANNED and NOT executable by the old pilot scripts as-is. The repository's existing developer runner is not a paper-scale trainer.

## Resources — planning estimates only
Compute reference 6*P*T FLOPs at optimistic ASSUMED end-to-end 200 TFLOP/s per H100: 340M/20B ~56.7 GPUh/run; 1.3B/100B ~1083 GPUh/run. 27*56.7+9*1083 ~11,280 H100 GPUh theoretical proxy before SG matrix-solve overhead, evaluations, nonoptimal model utilization and tuning. Full non-diagonal decay currently lacks optimized CUDA kernel; actual runtime CAN be far higher/unbounded until implemented and benchmarked. Plan >=80GB VRAM per worker, >=128GB RAM per worker, streamed data and several TB shared scratch/checkpoints.