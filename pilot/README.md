> **Paper-level experiments (one full training run per configuration, comparable published training budgets):** see [FULL_PAPER_EXPERIMENTS.md](FULL_PAPER_EXPERIMENTS.md).
> This is a complete research protocol, **not yet a runnable paper-scale training pipeline**. Existing short-run scripts do not implement it.

# Wuke Candidate 3 — Historical Gradient Geometry for Shared LM Heads

**Research status (2026-11-22)**: algorithm + tests pushed to this branch.
Six CPU mathematical checks of direction/magnitude and exact logit preservation
passed locally; the full pinned-upstream GPU run has NOT been executed.
Do not report any new NLL gains before running it.

This branch RETAINS the coauthor's output layer:
    w_i = own_i + shared_{group(i)}
It never replaces the LM Head with gradient surgery or another optimizer.
The NEW question is whether historical target-gradient direction, and then
log gradient magnitude, choose better shared-token groups than the original
dynamic k-means on current output weights.

**Read [METHOD.md](METHOD.md)** for the exact math, confidence threshold,
seven matched arms, limitations and the experimental decision rule.
**Read [GATE3.md](GATE3.md)** for the current Wuke funnel status.

## Quick start from the repo root

    python3 -m unittest discover -s pilot -p 'test_*.py'
    bash pilot/run_3h.sh --device cuda

Optional hyperparameters for the two positive-gradient arms:

    bash pilot/run_3h.sh --device cuda --positive-min-count 3 --magnitude-weight 0.35

Real pre-tokenized, disjoint FineWeb GPT-2 files:

    bash pilot/run_3h.sh --device cuda --real-data-dir /path/to/fineweb_bin

Python 3.10+, torch, numpy, git, Linux GNU timeout. The runner pins the
coauthor's MIT-licensed upstream repository commit to
d0dc052749a97061f6daa6907371bd0a2ec33405 and patches ONLY an
ignored local checkout. The original source is NOT changed in GitHub.

For real data supply separate source files (uint16 token IDs, 1024-byte
header): fineweb_train_000001.bin and fineweb_val_000000.bin, plus
optional counts.npy. No network data download is included. Synthetic
topic/Zipf fallback is NOT real-language-model evidence.

## First screen

Model: width32, layers2, heads2, V~6000, context64, batch32,
400 steps, seeds 0/1, regroup every 50 steps, 64 groups.

| Arm | Description |
| --- | --- |
| adam_eps | strong unshared output-only Adam epsilon baseline |
| fixed_random | shared output rows, random fixed groups |
| fixed_frequency | shared output rows, fixed frequency groups |
| learned_weights | coauthor's ORIGINAL dynamic output row k-means |
| learned_gradients | earlier Wuke full-gradient EMA Euclidean clustering |
| positive_direction | NEW positive-target history with spherical cosine k-means |
| positive_dir_mag | NEW positive-target direction + log-magnitude feature |

Tokens with fewer than three TRAIN target occurrences use effective output
weight centroids as a fallback; if <64 reliable tokens exist, all groups
fallback to original weight k-means. Cluster count is fixed at 64; individual
cluster sizes are automatically unequal. A group change only reparameterizes
existing effective output rows; current logits are unchanged (asserted).

Primary: held-out overall NLL and NLL excluding UNK, frequency-conditioned
NLL (1–9 / 10–31 / 32–99 hits), cluster sizes, reliable-word share,
directional conflict diagnostic, wall-clock. All baseline LR/schedules
match. Passing proxy-only is NEVER a GO. For the current full-paper study, train each configuration once with a larger authentic vocabulary and evaluate on large, independent held-out corpora; this 3h legacy runner does not implement that study.

Reports: .pilot_workspace_rare_v3/reports/report.json and decision.json,
with individual stdout/stderr files. The Linux launcher is wall-time bounded
to 3 hours excluding dependency installation; the runner's inner budget is
10500 seconds. No GPU success result has been claimed here.

Coauthor source and prior results:
https://github.com/vukrosic/llm-research-paper-drafts/tree/main/01-rare-words-shared-output-vectors
