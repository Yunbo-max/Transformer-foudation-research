# Project 11: Sparse Pointer (Block + Identity-Hash) × Online Jump-State Cache

**Status (2026-11-22):** implemented exploratory code, tests and a
self-contained synthetic simulator. No trained coauthor GPU or RTX benchmark
result is claimed. The coauthor 04+05 Identity + Pointer Doubling architecture
remains the methodological foundation.

## 1. Sparse Pointer Attention

Dense coauthor: A = softmax(causal(QK^T/sqrt(d) - no_backtrack)), P'=A P,
R'=A R, Q'=gate*A_Q P+(1-gate)*Q.

New block-routed option: divide keys into size-W contiguous blocks; score
their mean representations against the current pointer query; select up to G
COMPLETED historical blocks, refine their original token-level scores and
no-backtrack penalties, plus the active causal block. Renormalize softmax
over selected candidates only. Incomplete or future blocks NEVER contribute
to the routing score, ensuring causal index use.

Reference dense: O(Q*N*d) score work. Block-routed: O(Q*(ceil(N/W)+
(G+1)*W)*d) operations plus candidate gathering, before non-Pointer heads.
This is subquadratic when W scales with sqrt(N), G fixed, Q~N. Block mean
can miss rare keys: no recall bound is claimed. The reference implementation
is PyTorch, NOT a fused optimized CUDA kernel.

Controls: S0 dense; S-DenseTopK full-score-then-topk (still quadratic);
S1 completed-block mean routing; S2 fixed random-hyperplane identity-hash
routing. The latter indexes keys by signed projection bits, probes exact
and nearby Hamming-one buckets, and refines ONLY causal keys with original
QK + no-backtrack scores. It never uses future keys to route past queries.
Hashing is prior-art ANN, not a newly invented scheme. With a soft mixed
Pointer query the correct signature may be missed: no recall guarantee.
Code: pointer_accel.py and pointer_hash.py.

## 2. Online Multi-Level Jump-State Cache

Functional graph f, levels J0(x)=f(x),
J_{l+1}(x)=J_l(J_l(x)). Cache objects (node,level) store destination after
2^level hops. A query for k hops composes only its binary set-bit levels.
Base successor facts remain available; the cache stores composed states.

Budget: sum of BF16 state-equivalent bytes <= B (logical capacity only).
Policies C0 none/recompute, C1 LRU, C2 LFU, C3 random, C4 level-aware LRU,
C5 Multi-Level Jump-State Utility. Utility uses only past requests:

 U_t(x,l) = (1+log(1+history_count(x,l))) * min(2^l,512)
            / (1+age_t(x,l)/64).

This is a candidate heuristic, NOT the OpenAI k-Server algorithm or any
competitive-ratio theorem. It can be worse than LFU or LRU, and that must be
reported. Updates to base facts conservatively invalidate the whole cache;
selective dependency repair has NOT been implemented.

The simulator stores integer destinations, not real GPU KV vectors; it
reports estimated BF16 state-equivalent bytes *excluding metadata*. Don't
claim measured VRAM savings from the cache study. Source: jump_cache.py.

## 3. Required 3x2 independent/crossed ablation

| | No cache | Multi-Level Utility cache |
|---|---|---|
| Dense Pointer | S0C0 | S0C1 |
| Block Sparse Pointer | S1C0 | S1C1 |
| Identity-Hash Sparse Pointer | S2C0 | S2C1 |

All arms receive identical generated facts and queries. The combined study
uses a frozen random-embedding Identity-style edge reader, and the Jump
Cache composes the retrieved values. Wrong sparse one-hop reads may corrupt
later cached jumps: no oracle corrections are inserted. Measure end-to-end
query accuracy, simulated edge reads and compositions, candidate score
counts, actual CPU/GPU elapsed time, cache entries and cache hit rate.

Independent cache comparisons use fixed traces and 12.5/25/50% of the
number of nodes as slot budgets, together with nonstationary Zipf-like
request popularity and optional edge edits. Source: run_accel.py.

## 4. Optional actual coauthor model training integration

integrate_pointer.py is a fail-closed patcher for pinned coauthor pdg.py
(commit d0dc052749a97061f6daa6907371bd0a2ec33405). It changes ONLY
internal two Pointer heads' P/Q/R reads for --ptr=11 when the new
--wuke_sparse_mode block or hash flag is set. Dense fallback remains intact.

The *ordinary Transformer attention and original dense no-backtrack
residual attention are NOT sparsified*, so this is an accuracy feasibility
pilot, NOT a proven full-model CUDA acceleration or memory reduction.

run_sparse_model.py compares original dense vs block2/block4/hash at 50
steps, paired seeds 0/1, on reverse7, reverse5, repeated7, random7.
Records final answer accuracy, runtime and score-pair counts.
GPU and library integration are NOT verified in this authoring session.

## 5. Reproduction and 3-hour bound

    python3 -m unittest discover -s pilot -p 'test_*.py'
    python3 pilot/run_accel.py --suite all --device cpu
    python3 pilot/run_sparse_model.py --device cuda --steps 50
    bash pilot/run_accel_3h.sh cuda

The GNU timeout launcher caps the WHOLE attempt to 3h, including the
pinned source clone. Results go in ignored .pilot_workspace_pointer_* trees.

## 6. Local CPU proxy observations (NOT official GPU results)

Using the standalone PyTorch/reference Python algorithms in this research
session (seed 11, d=32, N=512, block_size=16, 2 routed blocks, 8 hash bits,
2 hash probes, candidate bucket 32, local window 8), the measured
synthetic Identity needle accuracy / exact QK comparison pairs were:

| Sparse mode | Needle accuracy | Exact QK pairs |
| --- | ---: | ---: |
| Dense | 1.000 | 262,144 |
| Block Mean | 0.654 | 19,968 |
| Hash | 1.000 | 5,740 |

These are ONE-SEED controlled synthetic findings, not trained multi-hop
coauthor accuracy. CPU operator latency for Hash was WORSE than dense.
No full-model or actual GPU memory/latency improvement is established.

On a deliberately harder cold/hot mixed online trace with 1024 nodes,
512 queries and capacity 128 slots, this untrained Utility admission
policy saved ~55.0% of symbolic work, whereas LFU saved ~57.4%.
Thus our Utility variant does NOT yet beat strong baselines, and its
heuristic should not be claimed as successful. The next research
iteration must improve this comparison *without tuning on held-out
evaluation traces*. The code includes the harder cold-request mixture.

## 7. Acceptance gate


The new mechanism is research-worthy only if: (i) model sparse Pointer
accuracy remains within 3 percentage points of dense on both paired
reverse7 seeds, (ii) larger-context *measured* GPU latency and memory improve
after kernel work, not just theoretical score counts, (iii) utility cache
beats both LRU and LFU at equal capacity on more than one shifted-query
distribution, and (iv) at least one sparse+cache arm outperforms both single-axis
alternatives under a matched accuracy and memory budget. Negative results are legitimate; proxy-only
success is never publication evidence.

### Prior art and originality
- Coauthor 04/05:
  https://github.com/vukrosic/llm-research-paper-drafts
- AttentionIndex (Nov 2026): https://arxiv.org/abs/2611.02508
- Routed Memory (ICML 2026): https://proceedings.mlr.press/v306/zeng26h.html
- Depth-Cache tradeoff: https://arxiv.org/abs/2604.17935
- k-server is online resource allocation theory, NOT an implementation:
  https://github.com/openai/math

Only the PARTICULAR coupling of coauthor trainable Pointer Doubling with
computationally valued per-level cached states and sparse Pointer routing
is our current hypothesis. Its novelty and actual benefit remain unproven.
