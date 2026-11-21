# SG-Delta developmental gate: retention and new-fact learning

## Current status — 2026-11-21

The main method remains historical-Query joint directional decay and protected
writing. Predicted-Skip remains a separate branch. This repaired revision is
source-reviewed; numerical checks and training on its exact files are pending.
Incoming history reported earlier CPU algebra/smoke checks; those reports are
not verification of this modified implementation.

Read LOCAL_AGENT_RUNBOOK.md for exact target-machine steps. Large profiles are
sizing-only; no real-model benefit or kernel speedup is established.

## Preserved scientific controls

1. Initial facts receive equal write exposure. Query accesses are skewed, with
   guaranteed hot and cold probes after interference.
2. Every token also creates learned K/V projections. Repeated query tokens
   therefore change hidden Key/write statistics: the neural task does not hold
   them fixed. The controlled mechanism script fixes statistics explicitly.
3. Genuinely new associations and a new-key/same-value alias are tested.
   Retention alone is insufficient when new associations cannot be learned.
4. Paired arms share examples, initialization seed, training steps, architecture,
   precision, optimizer and evaluation protocol.

## Arms

| Arm | Write | Decay | Role |
| --- | --- | --- | --- |
| isotropic | isotropic Delta direction | scalar | simple baseline |
| query-write | historical Query geometry | scalar | write ablation |
| query-decay | isotropic Delta direction | historical Query geometry | decay ablation |
| query-joint | historical Query geometry | historical Query geometry | main method |
| key-joint | historical Key geometry | historical Key geometry | same matrix capacity |
| channel-decay | isotropic Delta direction | learned diagonal | diagonal comparator |
| reverse-joint | historical Query geometry | complementary protection | optional negative control |

## Current screen

The executable rule is in screening.py and run.py. Every requested arm and seed
must finish; at least two distinct training seeds and 256 paired held-out
sequences per panel are required. Both baseline and candidate old-post accuracy
must exceed max(1/pairs, observed hot-target fraction) by five percentage points,
with a minimum of 15%; both must also reach 15% genuinely new-fact accuracy.
This blocks a simple hot-value-only strategy, not every possible shortcut.

Within each seed, query-joint must improve old-post recall by at least three
points over key-joint and two points over isotropic, channel-decay, query-write
and query-decay, with a positive lower paired sequence-bootstrap bound. Alias
and new-fact lower bounds must remain at least minus three points against
isotropic and key-joint. Hot/cold subgroup counts and accuracy are descriptive;
cold means all nonzero fact indices, not necessarily previously unqueried facts.

Reports are INCONCLUSIVE, NO_CLEAR_DEVELOPMENTAL_ADVANTAGE or
PROMISING_DEVELOPMENTAL. None is a native benchmark or scientific PASS.
Intervals condition on each trained seed and do not establish training-seed
population uncertainty or correct for arbitrary experiment selection.

## Interpretation and later work

Controlled algebra exhibits tradeoffs; it does not establish that joint decay
improves over write-only, or that query frequency identifies semantic importance.
Allow those hypotheses to fail in evaluation. Old numeric outputs using fast
retention .98 do not apply to the revised mechanism script's .95.

After target-machine correctness checks, integrate native benchmarks and
faithful author baselines. Full non-diagonal SG decay is not directly supported
by GDN-2's channel-wise kernel. Larger experiments require an equivalent
efficient implementation and measured resource budgets.

