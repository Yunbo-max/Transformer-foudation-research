# Paper and author-code review

Review scope: source-guided repair of the two existing methods, not a new
method-discovery batch or a claim of empirical superiority. Sources below were
read at the pinned revisions. No upstream implementation is silently replaced
by a weaker recreation and called an official baseline.

## 1. Preconditioned DeltaNet (PGDN)

- Paper: https://arxiv.org/abs/2604.21100
- Author repository revision: 7bd753279af87b39114149a104c5bde9bf67145f
- Actual reference recurrence:
  https://github.com/ntumm120/preconditioned-deltanet/blob/7bd753279af87b39114149a104c5bde9bf67145f/3rdparty/flash-linear-attention/fla/ops/precond_gated_delta_rule/naive.py
- Layer:
  https://github.com/ntumm120/preconditioned-deltanet/blob/7bd753279af87b39114149a104c5bde9bf67145f/3rdparty/flash-linear-attention/fla/layers/precond_gated_deltanet.py

The readable implementation accumulates recurrence/statistics in FP32, reads
the residual with the original key, and writes with a preconditioned key.
Its practical diagonal preconditioner has learned update/retention, a bounded
log-space squash and scaling; it is not just the paper's exact RLS denominator.
The SG repair borrows numeric/reference organization, while preserving SG's
query-based metric and full directional decay. No PGDN kernel equivalence is
claimed for full non-diagonal SG decay.

## 2. OSDN

- Paper: https://arxiv.org/abs/2605.13473
- Author repository revision: 39406acc683a562f8a4aa303daaffd3ddcec4db8
- Reference:
  https://github.com/Lhongpei/OSDN/blob/39406acc683a562f8a4aa303daaffd3ddcec4db8/src/osdn/reference.py

The reference makes pre-update versus post-update preconditioner timing
explicit, uses original keys for residual reads and scaled keys for writes,
and accumulates in FP32. Its APF forgets preconditioner calibration, not the
same quantity as SG's directional decay of the content state. The repair
adopts explicit helper interfaces and timing tests, not OSDN's update rule.

## 3. NVIDIA Gated DeltaNet-2

- Paper: https://arxiv.org/abs/2605.22791
- Author repository revision: de8476de0fecae08fc16f0bf3154900244c61c73
- Operator:
  https://github.com/NVlabs/GatedDeltaNet-2/blob/de8476de0fecae08fc16f0bf3154900244c61c73/lit_gpt/gdn2_ops/chunk_gdn2.py
- Layer:
  https://github.com/NVlabs/GatedDeltaNet-2/blob/de8476de0fecae08fc16f0bf3154900244c61c73/lit_gpt/gdn2.py

Default operator behavior includes a conditional erase-contrast beta guard,
normalized Q/K, FP32 recurrent state and output scale K**(-1/2). The guard
uses straight-through erase-gate gradients. It is not equivalent to normal
autodiff through its projection. Short causal Q/K/V convolutions are enabled
in the full author layer.

Our small decoder is still not the full NVIDIA architecture. Default-on
guard behavior and raw recurrence algebra are separate validation cases.
Intermediate chunk states exposed by the author operator are inference-only;
training here uses sequential external chunk calls with final-state gradients.
All paired arms use the same dispatch. Soft suppression saves no kernels by
itself. CUDA parity and wall-clock profiling remain required.

NVIDIA source is not vendored. Its repository has a non-commercial source
license; follow upstream terms for a separately installed checkout. The small
PyTorch guard reference is written independently from the mathematical
condition and must be numerically compared with the pinned operator.

## 4. Vuk Rosic's Predicted-Skip draft and implementation

- Repository revision: d0dc052749a97061f6daa6907371bd0a2ec33405
- Manuscript:
  https://github.com/vukrosic/llm-research-paper-drafts/blob/d0dc052749a97061f6daa6907371bd0a2ec33405/03-linear-attention-skip-lookups/PAPER.md
- Actual code:
  https://github.com/vukrosic/llm-research-paper-drafts/blob/d0dc052749a97061f6daa6907371bd0a2ec33405/03-linear-attention-skip-lookups/mix.py
- Author results:
  https://github.com/vukrosic/llm-research-paper-drafts/blob/d0dc052749a97061f6daa6907371bd0a2ec33405/03-linear-attention-skip-lookups/RESULTS.md

The code ALREADY provides --lpsE 1, scaling both write and erase. Joint
suppression itself is therefore not newly introduced in this branch. Both
write-only and joint controls are credited to this source. Our work adapts
them to the exact channel-wise GDN-2 recurrence and checks the implementation.

In the pinned author's `code e` path, decay, write and erase gates are
per-token/head scalars: D=alpha*I, w=beta, b=beta*e. This wrapper instead uses
NVIDIA's key-channel decay, key-channel erase b and value-channel write w,
with separate projections. That is an integration difference, not a new
Predicted-Skip gate or an invention of channel-wise GDN-2. The small decoder,
optimizer and diagnostic protocol also differ, so its losses cannot directly
establish superiority over the author's reported runs.

The author's chunked state is differentiable through the predictor. This
wrapper's default gate_memory_detach=True changes only that derivative; its
alternative remains explicit. No benefit from this difference is assumed.

The author predictor uses learned strength per layer/head and optional causal
short convolutions. Its strength is --lps * exp(learned), with the learned
parameter initialized to zero and --lps defaulting to zero (disabled). This
pilot chooses an initial strength of 0.25; that is not a universal author
default. The repair exposes strength and convolution settings and matches arms.
Cosine is an approximate signal: same value under a new key, weak collinear
reads, stale boundary states, and continual decay are explicit limitations.
A residual veto would change the algorithm and its chunk dependencies, so it
is not silently substituted in this correctness repair.

## 5. QED comparison boundary

Primary paper: https://arxiv.org/abs/2608.13668

QED changes the erase-side direction using the current query. SG changes the
write direction and decay using historical query statistics. Predicted-Skip
changes gate amplitude. We did not locate a verifiable public author QED
implementation in this search, so no locally reconstructed code is described
as official QED and no QED performance comparison is claimed.

## Evaluation boundary

The inherited generator in common.py is a developer-created associative
recall diagnostic, including alias and new-fact targets. It is not the native
published MQAR generator/scorer, FineWeb, or RULER. Its paired report can guide
debugging but cannot establish scientific effectiveness or novelty.
Qualified official benchmark integration and result validation remain pending.
