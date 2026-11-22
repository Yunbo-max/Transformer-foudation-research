# Predicted-Skip on GDN-2 — FULL PAPER EXPERIMENT PROTOCOL (2026-11-22)

STATUS: COMMITTED DESIGN ONLY; the existing small developer script, unverified official-kernel parity, and current diagnostic are NOT paper-scale training/evaluation. Never claim skip gate saves GPU compute unless actual kernels skip real computation.

## Reference standard
- Official NVlabs/GatedDeltaNet-2 (arxiv 2605.22791): 1.3B / 100B FineWeb-Edu tokens, 4K input, AdamW peak LR 4e-4, 0.5M-token global batch, warmup 1B tokens, RULER + real retrieval + commonsense.
- QED (arxiv 2608.13668) is a direct recent query-informed memory-update alternative, 340M/15B. Include at matched budget where the architecture permits.
- Original Vuk Rosic Predicted-Skip includes both write-only and joint erase/write; those are NOT newly invented in this Wuke code. The research question is faithful, useful transfer to official GDN-2.

## Full training matrix (3 full independent seeds each)
- 340M, 20B tokens, 4K sequence, matched training shards: 5 arms x 3 seeds = 15 trainings: official GDN-2 no skip, skip WRITE only, skip JOINT erase/write, shuffled previous-query predictor, current-key/read control. Add a source-faithful original Predicted-Skip and QED comparison as distinct, documented comparator if exact architecture normalization is possible; never label incompatible numbers apples-to-apples.
- 1.3B, 100B tokens, 4K sequence: 3 arms x 3 seeds = 9 trainings: GDN-2, write-only, joint. Fixed same tokenizer, optimizer recipe, gradients, data order and compute budget.
- VALIDATE official CUDA parity before training: compare unguarded and native beta guard, forward outputs, final states, Q/K/V gradients, all parameter gradients, chunk start snapshots; train/eval mixed-precision behavior must match. Unit tests are a necessary prerequisite, not sufficient evidence.

## Evaluations at EVERY completed seed
1. Natural held-out next-token NLL (>=20M nonoverlapping tokens); WikiText-103, LAMBADA, HellaSwag, ARC-C/E, PIQA, BoolQ, OpenBookQA and native real-world retrieval (SWDE, SQuAD, FDA, TriviaQA, NQ, DROP).
2. RULER 13 tasks x 4K/8K/16K/32K/64K where viable, official standard 500 test examples per task-length, PLUS >=2,000 each for preregistered S-NIAH-2/3, MK-NIAH and tracing critical settings.
3. Mechanism-specific tests: old-fact retention, answer replay/re-encoding, new-key/same-value alias, same-key replacement, changed answer, hot/cold skew, nonstationary topic shift, unrelated distractors. >=2,000 genuinely held-out sequences PER main condition and per model seed, never training contamination. Track false-skip on necessary association writes, edit suppression fraction and exact conditional accuracy, not only overall NLL.
4. Ablations: write-only vs joint, shuffled previous-query vs current query, native beta guard vs raw (separate configuration), gamma/threshold strength, boundary-state detachment, chunk size, long vs short contextual recurrence. Matched LR tuning for all arms, no cherry-picked gamma.
5. Real throughput: compare official GDN-2 kernel and our complete official-backend wrapper for training tokens/sec, prefill/decode token/sec, actual CUDA memory, synchronization, extra state reads, additional kernel calls at 4K-64K. A SOFT gate not skipping work is not a speedup.

## Non-negotiable rules (full-paper, not feasibility screening)
- Minimum THREE independently initialized TRAINING runs for EVERY main result, paired seeds {11,22,33} across arms with identical frozen data/order, tokenizer, optimizer family, train tokens and parameter budget. Report all three values, mean, standard deviation; preserve seeds and checkpoints. A bootstrap over questions NEVER substitutes for training seeds.
- Pre-register a single primary metric and contrasts. Freeze train/val/test disjointness and test prompts before running. Validation selects hyperparameters (per-arm equal tuning budget), never test. Report natural-distribution scores AND hard conditional groups; no post-hoc selective tables.
- Synthetic benchmark generators must use independently held-out graph instances/keys and fixed generation seeds common across methods; report sample counts in every table. Two-sided paired bootstrap on test instances (cluster by document/graph), seed-level variability separately, and appropriate multiplicity correction for many tests.
- Log wall clock, throughput, actual peak GPU allocated/reserved VRAM, FLOPs including routing/index construction, host RAM, model size, accuracy/NLL and checkpoints. Run on real NVIDIA GPUs. No CPU/symbolic cached-byte or pair-count proxy can be reported as measured GPU speedup.
- Use full training, never tiny 50-step or 400-step runs for claims. All stated experiments below are PLANNED and NOT executable by the old pilot scripts as-is. The repository's existing developer runner is not a paper-scale trainer.

## Resource calculation
At hypothetical 200 TFLOP/s sustained H100 effective compute, 340M/20B = 56.7 h/run and 1.3B/100B = 1083 h/run, giving (15*56.7)+(9*1083) ~10,600 H100 GPUh BEFORE actual implementation overhead, kernel validation, long-context evaluation, tuning and failures. 80GB H100 or 96GB NVIDIA GPU per full-scale worker; 128GB+ CPU RAM, fast sharded data, multi-TB scratch.