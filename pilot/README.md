# Pilot C — Muon Confidence-Based Hold (<= 3 hours)

Repository: https://github.com/Yunbo-max/Transformer-foudation-research
Reference: https://github.com/vukrosic/llm-research-paper-drafts/tree/main/02-muon-hypergradient-learning-rate
Pinned MIT upstream SHA: d0dc052749a97061f6daa6907371bd0a2ec33405.

## Gap and new hypothesis
The published experiment shows (i) fixed early Hold fails when the initial high
rate has not recovered, (ii) very short runs can need a different global rate,
and (iii) noisy late cosine can cause shrinkage. Here, compare the reference
fixed hold and its already reported noise-relative post-hold rule with an
experimental *confidence hold* that waits for persistent near-noise cosine
evidence. This is a screen of a hypothesis, NOT a proven new optimizer.

## Run
Python >=3.10, torch, numpy, git. On local CPU:
  pip install torch numpy
  python3 pilot/run.py --device cpu --budget-seconds 10500

For available GPU:
  python3 pilot/run.py --device cuda --budget-seconds 10500
For real LM evidence:
  python3 pilot/run.py --device cuda --real-data-dir /data/heldout-fineweb-bins

Default generated data is a deterministic *synthetic topic/Zipf task*, not
FineWeb. Real input expects separate GPT-2 tokenized uint16 train/validation
binaries with a 1024-byte header, as documented in the runner. No labels
or validation loss enter the adaptive controller.

## Strict design
Two configurations: 225 steps/width64/batch16 (known short-run issue);
900 steps/width128/batch4 with a 10x-high starting rate (slow recovery).
Two paired seeds: 0,1. Each configuration includes:
- a 3-rate fixed reference sweep (0.00125, 0.005, 0.02; best-of-three only);
- original hypergradient with fixed 30% hold;
- published optional post-hold 2-standard-deviation correction;
- candidate persistent-confidence stopping with patience 35.

Primary: held-out NTP NLL and loss spikes; secondary: LR multiplier trace,
time to freeze and divergence. Claim NO GO unless the confidence controller
improves the fixed hold by >=0.005 nats and is within 0.006 of best-of-three
fixed on both seeds in the hard configuration. A sweep is not a perfect oracle.
All incomplete cells = INCONCLUSIVE.

Output: .pilot_workspace/reports/report.json and decision.json plus per-case logs.
Hard child wall-clock cap 10,500 s (<3h) excluding setup/clone.
No experiments have been run in this authoring session.
Full publication needs 100M+ real language model, broad LR/schedule controls,
multi-seed confidence intervals, and independently held-out text.
