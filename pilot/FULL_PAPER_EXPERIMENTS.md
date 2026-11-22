# Predicted-Skip — ACTIVE SMALL-SCALE FULL-PRETRAINING STUDY (2026-11-22)

**SCOPE LOCK:** matched 340M decoder, 20B FineWeb-Edu training tokens, 4096-token training context, **ONE full run/arm (seed 42)**. 1.3B/100B study DEFERRED. This is paper-standard FULL training, not 3-hour pilot. Current source-reviewed reference code and official-kernel parity adapter are NOT yet validated at full size.

## Closest research standards
- Official NVIDIA GDN-2 (arxiv 2605.22791) has 1.3B/100B, 4K and strong downstream comparisons.
- QED (arxiv 2608.13668) studies query-derived forgetting at 340M/15B.
- Original Vuk Rosic predicted skip already tests write-only and joint erase/write; this branch tests a GDN-2-specific use, not an assertion of inventing joint suppression.

## EXACT full-training matrix — 5 jobs
All 340M, 20B, 4K, identical data order and matched architecture/optimizer schedule, seed 42:
1. Official GDN-2 (no skip)
2. Previous-query Write-Only Skip
3. Previous-query Joint Erase/Write Skip
4. Shuffled previous-query read (negative control)
5. Current-key/read control (information and extra-compute control)
Optional original coauthor/QED source-faithful experiments are NOT included in the five-job total; compare existing published results only when settings are explicitly labeled incomparable.

**Preflight not an experiment/pilot:** validate original NVlabs GDN-2 parity (raw/native beta guard), all Q/K/V and parameter gradients, final states, chunk-boundary state reads, mixed precision and inference/training correctness. If parity fails do not train and do not silently substitute the unfused reference operator.

## Formal evaluations for EACH of the 5 trained checkpoints
- **Primary:** old fact retention after answers re-enter the stream, together with correct necessary NEW association/alias/overwriting writes. 2,000 independently held-out sequences per predeclared key scenario; quantify false skip, suppression frequency, new-key/same-value, same-key replacement and old/new NLL and accuracy.
- **Natural text:** FineWeb-Edu held-out NLL >=20M disjoint tokens; WikiText-103, LAMBADA, HellaSwag, ARC-E/C, PIQA, BoolQ, OpenBookQA, retrieval QA with SQuAD/TriviaQA/NQ/DROP/FDA under matching conditions.
- **Long-context:** official RULER 13 tasks at 4K/8K/16K/32K with 500 official items per task/length when feasible, and 2,000 additional independently held-out high-priority NIAH/trace examples at target lengths. 64K is conditional on memory/kernel, not guaranteed; record length failures.
- **Mechanism:** compare matched 5 trained rows; log gate rho by event, guard-induced differences, behavior under hot/cold drift, valid association edits and context size. Additional gamma/stop-gradient retraining studies beyond these 5 cells are deferred, not hidden in resource totals.
- **Efficiency:** measure actual official-backend 340M forward/backward training tokens/sec, prefill/decode wall latency, VRAM allocated/reserved and extra state read/kernel launches. A soft skip that still performs full compute cannot be claimed as GPU FLOPs saved.

## Non-negotiable execution and evaluation rules
- This is **complete small-scale pretraining**, not 50/400-step pilot: actually consume the declared token budget and save loss curves/checkpoints. One training seed (42) for each arm. Pair dataset shards, data order, tokenizer, initialization of comparable tensor shapes, token budget, architecture width/parameter count, optimizer/schedule and compute reporting. Never fabricate between-training-seed variance.
- Reserve validation for equal-budget hyperparameter selection BEFORE fixed final tests; define and freeze primary metric, held-out corpora, prompts and success criteria. Report all baselines, even worse ones. Held-out item bootstrap or document-cluster bootstrap concerns TEST sample uncertainty only; it does not establish training-seed robustness.
- Each official benchmark must use the official prompt/evaluation code and labeled native sample counts; custom supplements are reported in separate columns. Disjoint tokenizer train/test corpora; for synthetic graph tasks disjoint instances, held-out key inventories and graph structures. Avoid contamination.
- Report full-training wall time, effective tokens/sec, FLOPs inclusive of extra operators, model parameters, peak GPU **allocated/reserved** VRAM, host RAM, checkpoint footprints, prefill/decode median/P95 and test accuracy/NLL. All data pipelines must be sharded and reproducible.
- All rows below are **PLANNED**. Existing pilot scripts **DO NOT** implement the 125M/340M 5B/20B full-scale training pipelines. No complete GPU result is claimed. This stage has one model size only; larger scales and any extra training-required sensitivities are **DEFERRED**, not silently included in the time total.

## H100 80GB budget (estimated, not benchmarked)
340M/20B ~56.7 theoretical H100 GPUh/run at assumed 200TFLOP/s; five full jobs = **~283 ideal H100 GPUh** core. Budget **~480–1,200 H100 GPUh** including full validation and uncertainty in actual sustained throughput, conditional on official kernel integration. 80GB GPU, >=128GB RAM, ~1TB fast scratch/streamed data. Excludes full LR tuning and deferred 1.3B study.
