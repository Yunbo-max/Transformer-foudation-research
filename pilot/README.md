# Pilot A — Query-Protected GDN-2 (<= 3 hours)

This is a strict algorithm feasibility pilot, NOT evidence that an LM is trained.

## Main gap
Does a statistic of previously USED queries protect readouts better than
key-frequency statistics at MATCHED accuracy on newly written facts?
Does suppressing an answer write differ from changing the direction of an
interfering write? These must be isolated before adding a complex Branch/Fork.

## Run
  pip install torch
  python3 pilot/run.py --budget-seconds 10500

The default math screen uses the exact one-head scalar-gate, zero-decay special
case of the GDN-2 equation. No Triton is needed, so the diagnostic works on CPU
in minutes. It includes 3 fixed seeds with genuinely varying conflict difficulty and identical write streams per arm.
An aligned-Query/Key negative control must yield equal scalar-gate results.
It evaluates:
- plain rank-one GDN-2 update, scalar key-aware gate, scalar query-aware gate;
- query-aware Full Geometry direction, and key-aware Full Geometry control;
- repeated-answer Oracle: plain write vs forced skip vs previous-read similarity;
- old readout error AND new-fact error, with multiple matched-plasticity cutoffs.

DO NOT label the reference experiment as official 80M/350M GDN-2 training.

## Optional official kernel smoke test
Install the official NVIDIA repo and its stated PyTorch/FLA/Triton dependencies
in an existing GPU environment, without copying or modifying its kernels.
  git clone https://github.com/NVlabs/GatedDeltaNet-2.git
  git -C GatedDeltaNet-2 checkout de8476de0fecae08fc16f0bf3154900244c61c73
  python3 pilot/run.py --official-root ./GatedDeltaNet-2

The optional check imports unchanged official chunk_gdn2 and tests one
nonzero edit with all other token gates zeroed. BLOCKED is reported if official
dependencies or CUDA are unavailable, never silently replaced with a fake pass.
The parity check does NOT constitute a trained model comparison.

## Decision (predeclared)
At old-vs-new Pareto-matched thresholds, query scalar gate should beat key
scalar gate by >0.02 old-readout MSE (lower is better) if we want to proceed
with an official GDN-2 gate-only algorithm. Otherwise, do NOT claim that
query-conditioned scaling is superior merely because it writes less.
If only Full Geometry directions improve the tradeoff, mark
DIRECTION_MATTERS_GATE_ONLY_NOT_SUFFICIENT and revisit native recurrence
compatibility. Genuine full query geometry is not in general equivalent
to editing official GDN-2 b/w gates.

This is the small mathematical screen only. If it passes, the NEXT research
step is a controlled trained 80M official GDN-2 and a 350M validation, with
fixed held-out MQAR, RULER, LM NLL, throughput and memory across >=3 seeds.
No quality improvement is claimed until those runs actually complete.

## Provenance / output
- Existing Branch math: https://github.com/Yunbo-max/branch
  pinned comparison source SHA 287d5ebb30c6792d08e60fccda574c5b5fe1eda1.
- NVIDIA official GDN2: https://github.com/NVlabs/GatedDeltaNet-2
  pinned optional kernel SHA de8476de0fecae08fc16f0bf3154900244c61c73.
- Prior Predicted-Skip: https://github.com/vukrosic/llm-research-paper-drafts
  (coauthor's publicly released MIT draft; method is not claimed as ours).
- Results: .pilot_workspace/branch_reports/results.json. No results prefilled.
- The script has a 10,500-s experiment cap (<3h); dependency install/setup is
  separate and should be bounded by an outer job scheduler as appropriate.
