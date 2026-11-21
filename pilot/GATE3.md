# Wuke Funnel — Candidate 3: Dynamic Rare-Word Output Sharing

Status: NEXT RESEARCH CANDIDATE. The earlier SG-Delta and Predicted-Skip
branches have entered their own experiment-validation stage. Neither has been
declared empirically superior on official GPU benchmarks. This note concerns
only the third direction and does not launch GPU work.

## What is already established by the reference draft?

Source: Vuk Rosic, Helping Rare Words in Language Models, MIT reference
https://github.com/vukrosic/llm-research-paper-drafts/tree/main/01-rare-words-shared-output-vectors
Pinned SHA: d0dc052749a97061f6daa6907371bd0a2ec33405

Output weight: R[word] = Own[word] + C[cluster(word)].
Every 50 steps, k-means over current effective rows and a function-preserving
reparametrization keep all existing logits unchanged. In the source's SMALL
FineWeb-derived model, at V=6000 and 400 steps, tuned head-only Adam epsilon
achieves NLL 4.9931 and cluster+epsilon reaches 4.9720 (0.021 nats).
Across different vocab/length settings reported gains vs tuned epsilon are
roughly 0.016-0.039 nats. These are the coauthor's PREVIOUS results,
not new Wuke runs.

The source explicitly reports:
- Random/frequency-fixed clusters do not reproduce the learned clustering gain.
- Very rare words seen <3 times can get WORSE; 10+ occurrence cohorts benefit.
- A large 128-sequence batch reduces the gain.
- Some apparent gain can be a small-vocabulary UNK-token artefact.
- The PAPER.md pseudocode mentions resetting cluster optimizer state, while
  grid.py's actual recluster() currently DOES NOT reset Adam moments.
  Do not claim those two descriptions are already identical.

## Wuke's candidate NEW contribution

Hypothesis: the useful words to share an OUTPUT UPDATE with are better
identified by a recent gradient-history similarity than by the current
effective output-weight similarity. Existing weight k-means remains the
strongest learned-cluster control, not our new invention.

The local-only runner clones that exact upstream commit, adds a head gradient
EMA, optionally builds k-means on that EMA, and preserves effective output
weights across assignment changes. Current first screen compares:
  E: head-only Adam epsilon 1e-5 (strong optimizer control)
  R: random fixed groups, same group capacity
  F: frequency fixed groups, same group capacity
  W: original row-weight k-means re-cluster every 50 steps
  G: new gradient-EMA k-means re-cluster every 50 steps

All arms share model width32, 2 layers, vocab6000, 64-token context,
batch32, 400 steps, and paired seeds 0/1. Default source is a synthetic
topic-Zipf proxy, explicitly NOT FineWeb; you MUST run with disjoint
real FineWeb-GPT2 tokenized training/validation files to claim LM usefulness.
No model trained in the current GitHub authoring session.

## Evaluation

Primary: held-out NLL after REMOVING the UNK-token contribution;
original overall held-out NLL is also reported.
Controls: NLL on targets seen 10-31 times in train and its validation mass;
frequency bins, rare target mass, elapsed per cell, failures, trained model size.

GO for real replication only if gradient-EMA grouping beats weight k-means by
>=0.005 nats excluding UNK on both paired seeds, and improves over both
fixed-group controls. Weight-cluster baseline must itself beat head-only
Adam-epsilon by >=0.01 nats on both seeds. If only the original W arm works,
there is no demonstrated NEW gradient-grouping contribution.

Default synthetic proxy results are tagged INCONCLUSIVE_PROXY_ONLY regardless
of measured gain. The original published small-model improvement is not
sufficient for a NEW paper. Any missing/noisy runs are inconclusive.
Full next stage: token-disjoint FineWeb-Edu, larger vocabulary >=16K,
width 64/128, separate hyperparameter tuning, 3+ seeds, batch-size stress.

## How to run

From this Git branch:

    python3 -m unittest discover -s pilot -p 'test_*.py'
    bash pilot/run_3h.sh --device cuda

For REAL, disjoint uint16 GPT-2 token files (1024-byte header):
    bash pilot/run_3h.sh --device cuda --real-data-dir /path/to/tokenized_data

Outputs: .pilot_workspace_rare_v2/reports/report.json, decision.json,
per-case stdout/stderr logs. GNU/Linux 3-hour hard wall-clock launcher.
Clone / dependencies remain required. No GPU results currently verified.
