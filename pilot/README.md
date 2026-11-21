# Pilot B — Dynamic Shared Output (<= 3 hours)

**Next active Wuke funnel candidate**: read [GATE3.md](GATE3.md) for the
original paper's existing findings, our distinct gradient-history hypothesis,
UNK-excluded evidence requirements, and current unverified run status.

This branch does not modify main. It screens the MIT-licensed Vuk Rosic
reference: https://github.com/vukrosic/llm-research-paper-drafts/tree/main/01-rare-words-shared-output-vectors

Two claims: (1) function-preserving periodic regrouping beats tuned output-Adam
epsilon and fixed clusters; (2) clustering gradient EMA is more beneficial than
clustering the current output rows. Claim (2) is a new *hypothesis*, not a result.

## Execute

Preferred strict three-hour GNU/Linux command (includes checkout and data prep):
  bash pilot/run_3h.sh --device cuda


Requires Python >=3.10, torch, numpy and git. For a GPU:
  pip install torch numpy
  python3 pilot/run.py --device cuda --budget-seconds 10500

For real pre-tokenized training and separate held-out validation:
  python3 pilot/run.py --device cuda --real-data-dir /data/fineweb-bin

The source SHA is pinned to d0dc052749a97061f6daa6907371bd0a2ec33405.
The upstream is cloned, then patched only in an ignored local work directory.
Expected GPT-2 token files have a 1024-byte header and uint16 token IDs:
fineweb_train_000001.bin (or .slice), fineweb_val_000000.bin (or .slice).
An optional counts.npy is accepted, otherwise counts are computed. Data must
truly be disjoint. The default is a topic-Zipf *synthetic proxy* and is NOT a
FineWeb result. Vendor files remain licensed under their original MIT license.

## Prespecified comparison

Model: 2 layers, width 32, V 6,000, 64-token windows, batch 32, 400 steps.
Paired seeds 0,1. Arms: Adam-epsilon control, fixed random clustering,
fixed frequency clustering, dynamic row k-means, dynamic gradient-EMA k-means.
All runs must use identical training/validation sequences and learning rates.
The wrapper asserts that regrouping preserves the effective output head.

Primary scores: held-out token NLL **excluding UNK token**, overall NLL,
rare 10–31 occurrence bin and its validation share, rare target mass.
Move to real-data replication if dynamic rows beat Adam-epsilon by >=0.01 nats
in BOTH seeds and rare target mass is >=5%. Consider the gradient extension
only if it further beats row k-means by >=0.005 nats in BOTH seeds.
Incomplete runs are INCONCLUSIVE.

All outputs: .pilot_workspace_rare_v2/reports/{report.json,decision.json,*.log}.
There is a hard 10500-s subprocess time budget (under 3 hours), excluding
dependency setup and cloning. Do not publish synthetic results as LM gains.
At least 3 real-data seeds and a larger model are needed for publication.
No experiments were run when this branch was authored.
