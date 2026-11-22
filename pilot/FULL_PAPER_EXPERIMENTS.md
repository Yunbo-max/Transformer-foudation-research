# SG-Delta — ACTIVE SMALL-SCALE FULL-PRETRAINING STUDY (2026-11-22)

**SCOPE LOCK:** 340M decoder, 20B actual FineWeb-Edu training tokens, 4096-token train context, **ONE full run per arm / seed 42**. 1.3B/100B is explicitly DEFERRED pending results. This is a formal training study, NOT the old 3-hour pilot. Full trainer, true matrix-decay kernel, and native evals remain unimplemented.

## Closest empirical standards
- Gated DeltaNet-2 (NVlabs, arxiv 2605.22791): 1.3B/100B, 4K.
- Systematic Analysis of Hybrid Linear Attention (arxiv 2507.06457): 340M/20B.
- The Query Knows What to Forget / QED (arxiv 2608.13668): 340M/15B and 2 seeds.
Keep the nine-arm 340M study to establish the real mechanism rather than copying official paper's larger 1.3B spend.

## Mandatory FULL training — EXACTLY 9 jobs
Every row is a 340M **matched parameter/compute recipe** training on the SAME 20B token-order manifest, 4096 input context, fixed tokenizer, one seed 42:
1. GDN-2 official comparator
2. KDA official comparator
3. QED query-derived erase comparator
4. Isotropic Delta reference
5. Query-Write (write-only query geometry)
6. Query-Decay (decay-only query geometry)
7. Query-Joint (proposed full SG-Delta)
8. Key-Joint (equal-state-capacity control)
9. Channel-Decay (diagonal control)

Use a validation-only, equal-cost, predeclared tuning recipe. Model architecture/training family must be stated; report source-faithful baselines separately if the official configuration differs. Keep all convolution policies, token budgets and state-statistic bytes accounted for. No optional 1.3B row in this execution stage.

## Formal full-size evaluation of ALL NINE checkpoints
- **Primary:** retention **after interference** versus ability to learn genuinely new/changed facts. Independently generated held-out tasks: 2,000 queries per predeclared main condition, covering same-key overwrite, new-key/same-value, hot/cold query frequency, retained vs replaced facts, long distractors, no fact/key leakage. Report BOTH retention and plasticity; macro-average hard conditions and trade-off plot.
- **Natural text:** FineWeb-Edu held-out NLL on >=20M unseen tokens; WikiText-103 and LAMBADA; ARC-E/C, HellaSwag, PIQA, OpenBookQA, BoolQ, using published standardized protocols.
- **Memory and length:** native MQAR + RULER 13 tasks at 4K/8K/16K/32K where inference is supported (500 official samples/task/length; exact official configuration retained). Independently evaluate 2,000 examples at prioritized multi-key and variable tracing conditions. Long-length extrapolation is NOT guaranteed by 4K training; log OOMs and cap contexts transparently. Native retrieval QA (SQuAD, TriviaQA, NQ, DROP, FDA) using consistent prompt and documentation.
- **Mechanism:** same-shared-geometry arms above are the required training-based ablations; measure old/new selectivity, projected-query history, actual H and C=KxK costs, Cholesky operations, stability and learned decay. Further full-training hyperparameter sweeps NOT counted in 9 jobs are deferred.
- **Speed:** on the exact implemented full 340M model, measure train throughput, prefill/decode, GPU peak allocated/reserved and accuracy vs runtime. Do not treat unsupported non-diagonal decay as native GDN-2 acceleration.

## Non-negotiable execution and evaluation rules
- This is **complete small-scale pretraining**, not 50/400-step pilot: actually consume the declared token budget and save loss curves/checkpoints. One training seed (42) for each arm. Pair dataset shards, data order, tokenizer, initialization of comparable tensor shapes, token budget, architecture width/parameter count, optimizer/schedule and compute reporting. Never fabricate between-training-seed variance.
- Reserve validation for equal-budget hyperparameter selection BEFORE fixed final tests; define and freeze primary metric, held-out corpora, prompts and success criteria. Report all baselines, even worse ones. Held-out item bootstrap or document-cluster bootstrap concerns TEST sample uncertainty only; it does not establish training-seed robustness.
- Each official benchmark must use the official prompt/evaluation code and labeled native sample counts; custom supplements are reported in separate columns. Disjoint tokenizer train/test corpora; for synthetic graph tasks disjoint instances, held-out key inventories and graph structures. Avoid contamination.
- Report full-training wall time, effective tokens/sec, FLOPs inclusive of extra operators, model parameters, peak GPU **allocated/reserved** VRAM, host RAM, checkpoint footprints, prefill/decode median/P95 and test accuracy/NLL. All data pipelines must be sharded and reproducible.
- All rows below are **PLANNED**. Existing pilot scripts **DO NOT** implement the 125M/340M 5B/20B full-scale training pipelines. No complete GPU result is claimed. This stage has one model size only; larger scales and any extra training-required sensitivities are **DEFERRED**, not silently included in the time total.

## H100 80GB compute planning (not measured)
6*P*T/(200e12 FLOP/s) for 340M/20B gives **56.7 ideal H100 hours/job**, 9 jobs = **510 ideal H100 hours of core training**. Full paper-scale training+native evaluation reservation: **~1,050–3,550 H100 GPU hours** only AFTER an efficient non-diagonal SG kernel exists (estimate based on uncertain effective throughput + eval; NOT a validated upper bound). On current unfused matrix-solve PyTorch code no dependable finite runtime estimate exists. Provision >=80GB VRAM/worker, 128GB host RAM, >=1TB fast workspace and streamed data; measure actual peak before launch. Further 1.3B training and tuning not included.
