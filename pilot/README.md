# SG-Delta — Joint Historical-Query Write Protection + Directional Retention

Status (2026-11-21): PyTorch CPU smoke and math tests passed locally; GPU,
official GDN-2, real text and larger-model training NOT run. This is the sole
current SG-Delta main method. Previous query-write-only SG-Delta is an ablation.
There are no Fork/Replace, state banks or extra chunk-start reads.

## Exact algorithm (one recurrent state per head)

For each chunk c, C_q uses only already-observed queries:

    H_c = lambda * I + mu * C_q,c
    P_c = I - lambda * solve(H_c,I)
    D_c,t = alpha_fast,t * I + (alpha_slow,t-alpha_fast,t) * P_c
    S_before = D_c,t @ S_{t-1}
    e_t = v_t - S_before.T @ k_t
    u_t = solve(H_c,k_t)
    a_t = u_t/(tau + k_t.T @ u_t)
    S_t = S_before + outer(a_t,e_t)
    output_t = S_t.T @ q_t

Update C_q using a strict causal exponentially weighted q q^T after each
chunk; factor H once per chunk. The spectrum of P lies in [0,1] and D stays
between the two retention factors 0<alpha_fast<=alpha_slow<=1. Both protection
of the Write direction and slowed decay derive from the SAME historical
Query statistics. Current implementation counts actual Query occurrences
but does not yet infer semantic retrieval success or use future gold labels.
It is not a scalar beta gate and is not the old write-only model.

## Quick start

Python 3.10+, PyTorch with appropriate CUDA build when using NVIDIA GPU.

    python3 -m unittest discover -s pilot -p 'test_*.py'
    python3 pilot/run.py --profile smoke --methods isotropic query-joint --seeds 0
    bash pilot/run_3h.sh --device cuda

For CPU, use --device cpu. GNU/Linux three-hour launcher uses a 10,800-second
outside timeout and 10,200-second inner experimental budget. Report files:
runs/sgdelta_joint/manifest.json, per-arm seed JSON, and decision.json.
Non-completion is INCONCLUSIVE, never a win.

## Model sizing (measured by constructed model parameter counting, not training)

| Stage | Decoder / heads / key dim | Parameters | Status |
| --- | --- | ---: | --- |
| CPU smoke | D32 / L1 / H1 / K8 | 14,250 | locally smoke-tested |
| 3-hour pilot | D128 / L2 / H2 / K16 | 373,320 | locally smoke-tested |
| stress | D192 / L3 / H3 / K32 | prints at startup | not full-run validated |
| 100M-class sizing | D768 / L12 / H12 / K64 | 93,906,720 | sizing only |
| 340M-class sizing | D896 / L24 / H14 / K64 / vocab32K | 308,642,208 | sizing only |

Size only (no weight allocation):

    python3 pilot/run.py --profile 100m-sizing --methods query-joint --seeds 0 --dry-run
    python3 pilot/run.py --profile 340m-sizing --methods query-joint --seeds 0 --dry-run

Large profiles are BLOCKED from training in this reference: no optimized full
non-diagonal decay kernel exists here. Pilot: 1x 4090 24GB, CPU RAM >=16GB.
After GPU profiling: 1x H100 80GB or 97GB GPU for ~100M; use multiple H100
for large-scale 300M NTP if 10B+ tokens are required. These are recommended
budgets, not measured device peaks.

## Data and evaluation

Pilot data: seeded on-the-fly synthetic MQAR; distinct initial Key->Value
associations, frequently queried old facts, 4 new interfering facts, and an
additional NEW KEY with the SAME value as a previously retrieved fact.
All input sequences are identical across arms at the same seed, while
validation uses held-out generator seeds. No dataset download or gold labels
are supplied to the model. Train with causal next-token cross entropy masked
to answer positions only.

Default: 8 initial facts, 8 queries, 4 distractors, batch 8, 150 steps,
two seeds (0,1). Three evaluation panels: zero distractors, four distractors,
and doubled facts at fixed distractors. Primary scores: old fact recall before
and after interference, new alias-Key accuracy, answer NLL/accuracy. Secondary:
mean protection spectrum, write cosine, decay rates, GPU peak, wall time,
steps and tokens. Do NOT report artificial pilot as FineWeb or RULER.

Arms:
- isotropic: Delta Rule with scalar decay, no query geometry
- query-write: old SG-Delta full Write only, scalar decay
- query-decay: query-dependent full Decay only, isotropic Write
- query-joint: NEW PRIMARY SG-Delta (both)
- key-joint: same full matrix capacity, historical Key statistics instead
- channel-decay: learned input-dependent diagonal decay baseline
- reverse-joint: negative control that slows unimportant directions

Predeclared heuristic for follow-up: two complete seeds; query-joint improves
post-interference old recall by >=3pp over key-joint and >=2pp over BOTH
query-write and query-decay; alias-key learning falls no more than 3pp vs
key-joint. Also report old-vs-new learning Pareto curves. This does NOT replace
confidence intervals or prove novelty; close-to-chance results are inconclusive.

## Stage 2 / 3 planned ONLY after pilot passes

1. Numerically exact forward/state/gradient parity versus a faster
   implementation. PGDN has a reusable asymmetric key kernel, but does NOT
   directly support our full non-diagonal Query-conditioned decay.
2. 93.9M text LM with document-disjoint streamed FineWeb-Edu; fixed tokenizer,
   100M to 1B tokens per model as an intermediate check, not within 3 hours.
3. 308.6M paper class: compare official GDN-2, PGDN, OSDN, Bayesian Memory at
   matched training tokens, state bytes, sequences and hardware.
4. Real evaluation: held-out LM NLL/PPL, JRT or Zebra-style recall, RULER
   MK-NIAH (2K/4K/8K/16K), interference curves, new-fact learning, training
   tokens/sec and peak VRAM.

References:
https://github.com/NVlabs/GatedDeltaNet-2
https://github.com/ntumm120/preconditioned-deltanet
https://github.com/Lhongpei/OSDN
https://github.com/NVIDIA/RULER

NVIDIA GDN-2 is subject to its NVIDIA Source Code License-NC and is NOT
redistributed here. No official GPU or paper-sized results have been obtained.
