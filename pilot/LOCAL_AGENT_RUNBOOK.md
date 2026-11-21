# Local validation runbook — SG-Delta

This revision is source-reviewed. Revised numerical checks and model runs have
not been executed by the Web reviewer. Execute them in the target environment.

## 1. Pin the checkout and environment

Use branch pilot/sgdelta-joint-query-decay-write with a clean tracked tree and
an existing compatible Python 3.10+/PyTorch environment. Record git rev-parse
HEAD, Python/PyTorch versions, device and exact commands. Earlier smoke records
do not validate this repaired revision.

## 2. Numerical and software checks

Run from the repository root:

```bash
python3 -m unittest discover -s pilot -p 'test_*.py'
python3 pilot/mechanism_check.py
python3 pilot/run.py --profile smoke --methods isotropic query-joint --seeds 0 --eval-batches 1 --eval-batch-size 2 --budget-seconds 120
```

Tests cover direct objective solves, off-diagonal geometry, limiting cases,
state precision, causality, gradients and reporting failure cases. The mechanism
script checks a controlled two-dimensional case, not end-to-end learning.
A one-seed smoke is scientifically INCONCLUSIVE by design.

Retain any failures and fix their cause before training. Do not loosen numerical
assertions merely to obtain a passing record.

## 3. Bounded developer diagnostic

```bash
bash pilot/run_3h.sh --device cuda --precision float32 --seeds 0 1
```

Every arm must use identical architecture, convolution, precision, geometry
scaling, data seeds and optimizer settings. Defaults use convolution size four
and raw geometry scale; the convolution differs from the previous runner.
--conv-size 0 selects the old architectural ablation. Dimension scaling is a
separate configuration and requires rerunning every paired arm.

The output directory is timestamped and printed. Preserve its manifest,
arm/seed JSON, decision and errors. Check completed steps, paired evaluation
seeds, subgroup counts, measured time and memory. Timeouts or missing requested
controls remain INCONCLUSIVE. Completed runs may still be unlearned or show no
advantage; preserve that outcome.

## 4. Return evidence and respect scope

Return exact revision, environment, commands, full test output, mechanism JSON
and untouched run directory. Numerical tests verify only their stated properties.
This generator is not official MQAR/FineWeb/RULER. Its result cannot establish
novelty, real-language-model benefit or speedup.

Before scaling, integrate qualified native generators/scorers and author
baselines, and verify an efficient implementation against this reference.
Large sizing profiles remain blocked from training.
