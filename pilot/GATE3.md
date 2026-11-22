# Wuke Funnel — Candidate 3: Historical Gradient Dynamic Shared LM Output

## Current stage

The prior two candidates (joint query-geometry SG-Delta and joint-erase/write
Predicted-Skip) already entered their own experiment stage. This is a NEW
third research candidate. The target is not to rerun the coauthor's existing
weight-kmeans result but to investigate a distinct, mechanistic extension.

**As of 2026-11-22**: the new gradient direction + magnitude code and
numerical unit tests are in this Git branch; CPU algebraic unit tests passed
locally. NO new full 3h GPU pilot, official real-text experiment, or evidence
of lower NLL has been produced. Main repository branch is not changed.

## Established prior and new idea

Coauthor's prior LM Head: effective row w_i=u_i+c_{g(i)}.
The assignment g(i) is dynamic K-means on w_i; reassignments preserve current
logits exactly. This IS existing work; do not claim it is ours.

Their SMALL-model reported 6000-vocab/400-step result: output-only tuned
Adam epsilon 4.9931 vs weight-kmeans shared head + epsilon 4.9720 NLL
(0.021 nats). The prior report also notes much smaller benefit at larger
batch, negative transfer for ultra-rare tokens and potential UNK artefacts.

Wuke's new grouping variants:
- G1: full output gradient step-EMA and Euclidean k-means (previous pilot).
- G2: accumulate gradient (p_y-1)*h only when token y is the training target;
      average by target occurrence; normalized *direction* -> spherical K-means.
- G3: combine G2's direction with STANDARDIZED LOG MAGNITUDE, coefficient
      .35; Euclidean k-means on the augmented features.
- For fewer than three training target occurrences, assign a token using
  the nearest learned group's OUTPUT-WEIGHT centroid. If fewer than 64
  reliable tokens have appeared, fall back to ordinary weight k-means.

All variants keep w_i=u_i+c_g, 64 groups and the same total trainable
parameter structure. Group cardinalities may vary. The new design adds
non-gradient buffers of O(Vd), not a second set of trainable output vectors.
No learned selective gate or adaptive number K is being claimed.

## Critical test of novelty and effectiveness

Competitors: coauthor dynamic weight k-means, strong head-only Adam epsilon,
fixed random and frequency groups, previous gradient-EMA k-means. Do NOT
compare only against ordinary unshared Adam.

Data: REAL disjoint FineWeb-GPT2 token files are required to make a language
modeling claim. Synthetic topic/Zipf data is explicitly a proxy. Pilot:
width32, depth2, 64 context, V~6000, 400 steps, paired seeds 0/1.

Metrics:
- Held-out NLL WITHOUT mapped UNK tokens (main);
- Full validation NLL, 1–9, 10–31, 32–99 target-occurrence cohorts;
- Gradient-reliable fraction, group size range, negative centroid-cosine
  fraction, train seconds and failed cases.

Tentative method screen only if in BOTH seeds the selected positive-gradient
method is >=0.005 nats better than coauthor weight-kmeans and better than
both fixed-group controls, while weight-kmeans itself beats tuned Adam
epsilon by >=0.01 nats and >=5% of validation mass is rare. Magnitude has
INDEPENDENT support only if G3 beats G2 by >=0.003 nats in BOTH seeds.
Even a positive real-text pilot is GO_TO_3PLUS_SEED_REAL_DATA_REPLICATION.
A synthetic positive is INCONCLUSIVE_PROXY_ONLY. No claim of first-ever
method or publication-ready results follows from this screen.

## Reproduce

    python3 -m unittest discover -s pilot -p 'test_*.py'
    bash pilot/run_3h.sh --device cuda
    # Optional genuine disjoint real text:
    bash pilot/run_3h.sh --device cuda --real-data-dir /data/fineweb-gpt2

Full details in [METHOD.md](METHOD.md) and [README.md](README.md).
Outputs under .pilot_workspace_rare_v3/reports/.
Pinned coauthor upstream code under MIT license remains unmodified in Git.
