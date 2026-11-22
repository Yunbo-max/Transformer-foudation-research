# Wuke Method: Target-Conditioned Historical Gradient Grouping for LM Heads

**Status:** implemented as a first-round testable *hypothesis*, not a reported
training improvement or a claim that no prior work exists. Based on the
coauthor's original function-preserving dynamic shared-output LM head.

## Retained output parameterisation (exactly the previous paper)

For each output token i, effective row R_i = U_i + C_{z_i}.
Logit for context h: logit_i = h^T R_i.
U_i is a token-specific learnable vector, C_g is a shared group vector,
z_i is the token->group assignment.

After ANY reassignment, compute the old effective rows R; set new shared
centres C'_g = mean_{i:z'_i=g} R_i, then U'_i = R_i - C'_{z'_i}.
This exactly preserves all existing logits. It does not preserve Adam states
or future optimization trajectories: use the same optimizer across arms.

## Research hypothesis (our NEW branch extension)

Original dynamic row grouping:
    z_i = KMeans(R_i)                      # coauthor's baseline, not our novelty

Existing simple pilot:
    m_i <- .95 m_i + .05 grad_{U_i} L      # includes negative-class gradients
    z_i = KMeans(m_i)                      # preliminary gradient-EMA control

New target-conditioned method:
    For each TRAIN example with target y:
       p_y = softmax(R h)_y
       g_y^+ = (p_y - 1)*h                 # correct-target output-row gradient
       G_y += g_y^+; n_y += 1

    a_i = G_i / max(n_i,1)                 # mean per target occurrence
    q_i = a_i / (||a_i||+eps)              # direction
    ell_i = clip( zscore(log(||a_i||+eps)), -3, 3 )   # amplitude

Direction-only arm:
    z_i = spherical_cosine_kmeans(q_i)    # normalized centroids

Direction + magnitude arm:
    phi_i = concat(q_i, eta*ell_i)
    z_i = euclidean_kmeans(phi_i), eta=.35

Reliability safeguard:
    Only cluster tokens with n_i>=3 and ||a_i||>eps by gradient history.
    Assign lower-count tokens by proximity of their existing EFFECTIVE
    output weight R_i to the learned groups' output-weight centroids.
    If fewer than K reliable words exist, fall back to original R_i kmeans.

The historical estimate is a PER-OCCURRENCE running mean, not the old
full-gradient PER-STEP EMA. It is collected during normal training forward
passes, detached from autograd, and NEVER accesses validation/test labels.
The count threshold is a baseline heuristic; it is not a confidence interval.

Why positive-target only? Negative-class softmax gradients p_i*h occur for
all absent tokens on nearly every step; ordinary full gradients can make
unrelated rare tokens share background update directions. We TEST whether
filtering them helps; it is NOT presumed to always be better.

This project does NOT yet implement learned per-token gates, adaptive
numbers of groups, differentiable routing, gradient surgery, or a second
copy of the full output table. K is fixed at 64 in the pilot; the sizes
of individual groups vary naturally.

## Seven matched pilot arms

| Arm | Original output representation | Group decision |
| --- | --- | --- |
| adam_eps | unshared R | strong head-only Adam epsilon=1e-5 |
| fixed_random | U+C[z] | fixed random groups |
| fixed_frequency | U+C[z] | fixed frequency groups |
| learned_weights | U+C[z] | original row-weight kmeans every 50 steps |
| learned_gradients | U+C[z] | full-gradient step EMA + euclidean kmeans |
| positive_direction | U+C[z] | NEW target-conditioned spherical cosine kmeans |
| positive_dir_mag | U+C[z] | NEW cosine direction + log-gradient magnitude |

All arms: width=32, depth=2, V~6000, context=64, batch=32,
400 steps, paired seeds 0/1, clusters K=64 (when applicable).
Default is on-the-fly synthetic Zipf/topic PROXY; NOT real text evidence.
The data are the same for paired arms. Optional source is genuinely disjoint
pre-tokenized FineWeb-GPT2 train+validation (see README for format).

## Gate and outputs

Main metrics: held-out mean next-token NLL EXCLUDING the UNK token, full NLL;
true 1-9 / 10-31 / 32-99 occurrences NLL cohorts; rare-mass fraction.
Also record group-size min/mean/max, fraction of reliable words,
directional conflict-to-group-centroid fraction, and elapsed training time.
A fixed K does not mean group sizes have to be equal.

The new direction or direction+magnitude arm must beat the original
learned_weights by >=0.005 nats on BOTH seeds excluding UNK, and beat BOTH
fixed-group controls. The original learned_weights should itself beat the
strong tuned output Adam epsilon by >=0.01 nats in both seeds with >=5% rare
target mass. Incremental magnitude benefit is reported separately; it must
be >=0.003 nats on both seeds for a tentative positive magnitude claim.
Proxy-only runs always report INCONCLUSIVE_PROXY_ONLY, regardless of gains.
Even a real 2-seed pilot only gates into 3+ seed replication, not a paper.

## Run on your local NVIDIA GPU (checkout this exact branch)

    python3 -m unittest discover -s pilot -p 'test_*.py'
    bash pilot/run_3h.sh --device cuda

    # With your separate, document-disjoint, GPT-2-pretokenized text shards:
    bash pilot/run_3h.sh --device cuda --real-data-dir /data/fineweb-gpt2

Results: .pilot_workspace_rare_v3/reports/report.json and decision.json.
Requirements: Python 3.10+, PyTorch, NumPy, git and GNU timeout on Linux.
One 24GB RTX 4090 is sufficient for this tiny model; 16GB RAM and 5GB free
workspace are sensible initial budgets, NOT measured peak usage.
No 100M+ experiment, real-text result or GPU execution is claimed.

## Novelty positioning

Do NOT claim that additive output vector sharing, dynamic regrouping,
gradient grouping, rare-token learning-rate adjustment, or cosine similarity
are new ideas: these have substantial prior work. The specific hypothesis
to verify is **target-conditioned historical gradient geometry guiding
function-preserving dynamic additive shared LM output embeddings**.
Before making first-of-its-kind claims, repeat a literature search at the
time of submission.

Existing reference:
https://github.com/vukrosic/llm-research-paper-drafts/tree/main/01-rare-words-shared-output-vectors
Relevant nearby analysis:
https://arxiv.org/abs/2609.37535
https://arxiv.org/abs/2603.10145
https://papers.neurips.cc/paper_files/paper/2023/file/5b6346a05a537d4cdb2f50323452a9fe-Paper-Conference.pdf
