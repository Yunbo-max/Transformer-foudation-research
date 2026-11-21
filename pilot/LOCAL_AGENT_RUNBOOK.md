# Local validation runbook — Predicted-Skip

This revision is source-reviewed. Revised unit tests, CUDA parity and model
training remain unexecuted by the Web reviewer. Both write-only and joint
suppression originate in the cited author implementation.

## 1. Pin checkout and environment

Use pilot/predskip-joint-erase-write-gdn2 with a clean tracked tree and an
existing Python 3.10+/PyTorch environment. Record git rev-parse HEAD, environment,
hardware and commands. Old scalar-gamma results do not validate this configuration.

## 2. Reference checks

From the repository root:

```bash
python3 -m unittest discover -s pilot -p 'test_*.py'
python3 pilot/oracle_diagnostic.py
python3 pilot/run.py --profile smoke --methods baseline joint --seeds 0 --eval-batches 1 --eval-batch-size 2 --budget-seconds 120
```

Retain all failures. One-seed smoke is scientifically INCONCLUSIVE. The oracle
intentionally shows failure cases of cosine skipping; do not relabel them as
successful redundant-write detection.

## 3. Official parity before official training

Use the upstream installation instructions and dependencies at this revision:

```bash
git clone https://github.com/NVlabs/GatedDeltaNet-2.git
git -C GatedDeltaNet-2 checkout de8476de0fecae08fc16f0bf3154900244c61c73
python3 pilot/validate_official.py --official-root ./GatedDeltaNet-2 --precision float32 --guards raw native --hidden 128 --heads 2 --key-dim 16 --value-dim 16 --chunk 16 --conv-size 4 --gamma 0.25 --output runs/predskip_joint/parity.json
```

Inspect outputs, final states, all operator input gradients and every used
model parameter gradient. The receipt binds source and validator, actual FLA
normalization, upstream revision, runtime, guard, precision and configuration.
Failing or mismatched receipts block official training. Retain failures and
investigate; do not relax tolerances merely to obtain PASS.

Generate a new receipt after changing precision, convolution, dimensions,
chunking, gamma, detach policy, dependencies or device. BF16 and FP32 are
separate checks, as are raw and native guard policies.

## 4. Matched bounded diagnostic

After successful matching parity:

```bash
bash pilot/run_3h.sh --device cuda --backend official --official-root ./GatedDeltaNet-2 --guard native --precision float32 --parity-report runs/predskip_joint/parity.json --seeds 0 1
```

Alternatively, use the slower reference-only diagnostic:

```bash
bash pilot/run_3h.sh --device cuda --backend reference --guard native --precision float32 --seeds 0 1
```

All arms use the same dispatch and architecture. --diagnostics is reference-only
and collects active-edit quantities in evaluation. Soft suppression does not
itself avoid a kernel call or establish a speedup.

## 5. Return evidence

Return revision, environment, commands, test logs, oracle output, parity JSON
and the full printed run directory. Check completion, counts and paired seeds.
Missing requested controls or timeouts remain INCONCLUSIVE. Report hot/cold,
new-fact and alias learning. This generator cannot establish native benchmark,
real-text or full NVIDIA-model performance; those integrations remain pending.
