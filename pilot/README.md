> **CURRENT ACTIVE EXPERIMENT (2026-11-22): 340M / 20B tokens / 9 complete trainings; one seed 42 per method, REAL full pretraining, then formal large-sample evaluation.** All larger model-scale expansions are **DEFERRED** until these results are reviewed.
> Formal protocol: [FULL_PAPER_EXPERIMENTS.md](FULL_PAPER_EXPERIMENTS.md). Machine-readable job design: [SMALL_SCALE_STUDY.json](SMALL_SCALE_STUDY.json).
> **Implementation status:** design only. The legacy 3-hour scripts below are NOT the full pretraining pipeline and do NOT run the locked 5B/20B-token study. Real training infrastructure, native benchmarks and (where needed) GPU kernels remain to be implemented.

# SG-Delta — Joint Historical-Query Write Protection and Directional Decay

**Current revision: source-reviewed repair; numerical tests and training on
these exact bytes have not been executed in the Web review.** Historical
smoke-test statements applied to earlier revisions, not this one.

Start with [LOCAL_AGENT_RUNBOOK.md](LOCAL_AGENT_RUNBOOK.md).
The full derivation and mathematical scope are in [MATH.md](MATH.md).
The primary papers and pinned author code are in [SOURCES.md](SOURCES.md).
[WEB_HANDOFF.md](WEB_HANDOFF.md) records changed settings and pending checks.

## Method

For each head, S is the single content memory with shape K x V. C is a separate
K x K statistic and must be included in memory accounting. It contains only
completed chunks' exponentially weighted projected-query outer products.

    H = lambda I + mu_eff C
    P = I - lambda H^{-1}
    D_t = alpha_fast,t I + (alpha_slow,t - alpha_fast,t) P
    Sbar_t = D_t S_{t-1}
    e_t = v_t - Sbar_t^T k_t
    a_t = H^{-1} k_t / (tau + k_t^T H^{-1} k_t)
    S_t = Sbar_t + a_t e_t^T
    output_t = S_t^T q_t

The approved joint formula is preserved. Cholesky, covariance, residuals and
state accumulation now run in FP32 under mixed precision, or FP64 for explicit
mathematical checks. No silent diagonal jitter or eigenvalue clipping changes
the objective. Authored regression checks compare the helpers used by the
actual mixer with a direct solve of the unified quadratic objective;
execution of these checks remains pending.

The code counts all projected queries, not verified retrieval success or
semantic importance. Protection is soft; stale frequently used associations
can still impede new learning. Within-chunk fixed-H stability is documented;
no unrestricted global Euclidean contraction theorem is claimed.

## Configuration changes

- The runner uses a causal Q/K/V depthwise convolution of size 4 in **all
  arms**; --conv-size 0 restores the original no-convolution ablation.
  The constructor retains conv_size=0 for compatibility.
- Original geometry defaults remain lambda=1, mu=8, tau=1, EMA=0.98.
- --geometry-scale raw preserves mu_eff=mu. The explicit alternative
  --geometry-scale dimension uses mu_eff=mu*K and is recorded as a different
  configuration, not silently applied during model scaling.
- --differentiate-statistics enables gradients through historical Query
  statistics; default behavior keeps the prior stop-gradient choice.
- --precision float32 is the default. --precision bfloat16 uses autocast for
  projections while keeping recurrent mathematics in FP32.
- Stats include actual mean decay and content/statistic state element counts.

Changing convolution, scaling, precision or stop-gradient settings invalidates
direct comparison with an old run. Re-run all paired arms at the same settings.

## Local commands

Run from the repository root in an existing compatible Python/PyTorch environment.

    python3 -m unittest discover -s pilot -p 'test_*.py'
    python3 pilot/run.py --profile smoke --methods isotropic query-joint --seeds 0 --eval-batches 1 --eval-batch-size 2 --budget-seconds 120
    bash pilot/run_3h.sh --device cuda --precision float32 --seeds 0 1

The one-seed smoke run is a software check and always has an INCONCLUSIVE
scientific screen. The launcher enforces a 10,800-second outer wall limit and
10,200-second inner budget; completion of all arms is not guaranteed.

Outputs go under a new timestamped directory inside runs/sgdelta_joint/.
The directory is printed at startup. Old outputs are not overwritten.
Files include manifest.json, arm/seed JSON, decision.json and any error files.

## Arms and developer diagnostic

Arms: isotropic, query-write, query-decay, query-joint, key-joint,
channel-decay, and optional reverse-joint. Query-joint remains the main SG
method; query-write is an ablation. The isotropic arm is a normalized Delta
reference with fixed effective beta at the default settings, not official
GDN-2.

The inherited generated diagnostic is preserved, including the newly added
new_post targets. It is not official MQAR. Three panels contain no interference,
standard interference, and more stored pairs. Evaluation defaults to 16
batches x 16 independent sequences per panel. Paired arms share exact examples.
Per-sequence outcomes, counts, seeds, source hashes, timing and memory are saved.

The legacy diagnostic screener (not the current full-paper protocol) requires two complete seeds, established
old recall above the contextual-value baseline, complete matched evaluation, old-recall improvement over
the simple baseline AND individual components, and no clear loss of alias or
new-fact learning. Uncertainty uses paired sequence resampling within each
fitted seed. This does not estimate uncertainty across a population of training
seeds and is not a formal multiple-comparison confirmation procedure.

There is no automatic scientific GO/PASS. PROMISING_DEVELOPMENTAL means only
that native benchmark follow-up may be worth doing; incomplete, chance-level,
or insufficiently sampled comparisons are INCONCLUSIVE.

## Scale and outstanding work

Smoke D32/L1/H1/K8; pilot D128/L2/H2/K16; stress D192/L3/H3/K32.
Actual parameter counts are printed for the selected convolution/configuration.
Earlier hard-coded sizing numbers are historical and have been removed.

    python3 pilot/run.py --profile 100m-sizing --methods query-joint --dry-run
    python3 pilot/run.py --profile 340m-sizing --methods query-joint --dry-run

Large modes remain sizing-only. This full-matrix reference has no optimized
non-diagonal decay kernel or streaming cache. No new GPU peak, training speed,
real-text performance, or official baseline reproduction is claimed.
Native FineWeb/RULER/JRT-style evaluation and faithful PGDN/OSDN/GDN-2
comparators remain the next scientific work after local correctness checks.

## Preserved concurrent protocol additions

The latest hot/cold query-frequency skew, guaranteed hot/cold post-interference
probes, and genuinely new-fact targets are retained. All groups save raw counts
and per-sequence outcomes. Repeated query tokens also change learned K/V
streams, so these neural experiments do not hold hidden Key statistics fixed.

Because the hot value is queried much more often, vocabulary-uniform chance
is a misleading baseline. The screen requires both baseline and candidate old
recall to exceed max(1/pairs, observed hot-target fraction) by five percentage
points, and requires both to reach at least 15% new-fact accuracy. These are
conservative diagnostic prerequisites, not proof of key-value binding.
The full requested method-by-seed matrix must finish; missing controls,
unmatched examples, insufficient samples or timeouts remain INCONCLUSIVE.

Here hot means fact index 0; cold groups all other fact indices, with fact 1
forced as a final probe. Cold does not mean never previously queried.
