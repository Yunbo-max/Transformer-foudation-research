# Wuke Candidate 11 — Identity + Pointer Doubling: Novelty and Order-Generalization Gate

**Status (2026-11-22): existing coauthor baseline runner committed; no new
trained order-generalizing architecture; no GPU success result verified.**
Candidate 10 Adaptive Muon was abandoned. The earlier SG-Delta, Predicted-Skip
and output-head branches are separate and remain untouched.

## Existing literature cannot be repackaged as a new Wuke contribution

- Classic parallel pointer jumping / binary lifting: doubling is established.
- Sanford/Hsu/Telgarsky (ICML 2024): logarithmic-depth transformers for k-hop,
  including pointer-doubling behavior under particular training conditions:
  https://proceedings.mlr.press/v235/sanford24a.html
- Saparov et al. (ICLR 2025): trained transformers expand reachable graph sets
  exponentially with depth:
  https://proceedings.iclr.cc/paper_files/paper/2025/hash/28d43a64147bf94921041599efdf791d-Abstract-Conference.html
- Merrill/Sabharwal (NeurIPS 2025): log-depth transformers can express graph
  connectivity and related path operations:
  https://arxiv.org/abs/2503.03961
- Memory Mosaics (ICLR 2025) already contributes identity-style next-token
  matching: https://proceedings.iclr.cc/paper_files/paper/2025/hash/59c3ac496e6fe7be0c2c2b95014e2210-Abstract-Conference.html
- Coauthor's NOV 2026 draft has ALREADY combined P-doubling, doubled
  no-backtrack R, and a gated second Q pointer / binary lifting:
  https://github.com/vukrosic/llm-research-paper-drafts/tree/main/05-attention-pointer-doubling

Do NOT claim Wuke invented any of those 3 mechanisms.

## Basic math of coauthor's existing head

Let a_j(i) match the current pointer to a prior token. Instead of
retrieving current token embedding E_j:
   one-hop:  P_{l+1}(i) = sum_j a_j(i) * E_j
retrieve already-advanced pointer:
   doubling: P_{l+1}(i) = sum_j a_j(i) * P_l(j)
The jump doubles each layer. With L layers and a lifting gate, the learned
pointer can choose among 1+2+4+...+2^(L-1) total hops.

Coauthor's SHORT synthetic reverse-order evidence: 7 links, 3 layers,
about 0.979 accuracy (8 pairs; chance 0.125). In FORWARD order, 7/3
about 0.082; RANDOM block order, 7/3 about 0.056. A mirrored predecessor
head recovers 4 forward-order links / 3 layers at about 0.976, but reaches
only 2^(L-1) links and was not tested with all no-backtrack/lifting
mechanisms. These are COAUTHOR results, not our own.

One short FineWeb cell did NOT establish text benefits: tuned attention
3.8002 held-out loss; identity-head 3.7782; doubling 3.7835; full lifting
3.8098. These unscaled numbers are from the original paper only.

## What the existing pilot DOES

Runs existing coauthor 05-attention-pointer-doubling/pdg.py at pinned source
SHA d0dc052749a97061f6daa6907371bd0a2ec33405, on:
- reverse 7 hops, L3, plain/one-hop/doubling/lifting;
- forward 4 hops, L3, plain/full/predecessor doubling;
- random 7 hops, L3, plain/one-hop/full;
- reverse 5 hops, L3, variable-length gate test;
- repeat 7 hops with duplicate-key distractors.

Baseline uses 8 KV pairs, 64-wide/3-layer/4-head small model, 75 steps,
2 seeds, and 256 evaluation examples per case. Generates data locally,
no benchmark download; intended budget <=3h on single RTX 4090.

Current acceptance gate explicitly REQUIRES random-order accuracy >=0.80
and forward predecessor >=0.80, so this checks known limitations but does
NOT implement the missing random-order algorithm. Running the same code
cannot be presented as a novel Wuke extension.

## Next actual RESEARCH question (proposal; NOT IMPLEMENTED)

**Can order-agnostic pointer composition learn arbitrary-hop key-value chains
without requiring facts to be placed in reverse dependency order?**

It is not enough to simply add the coauthor's already-described mirrored
predecessor head: both order directions, adaptive chain length and handling
interleaved facts should work in one model. A potential non-causal
read-only prefix encoder or query-conditioned scratchpad enables
composition among arbitrary source positions; its additional passes and
latency must be counted. Do NOT claim ordinary causal attention supports
full backward reuse of pointers from future token positions. A changed
prefix-attention architecture is a new compute assumption, not free speed.

Require matched:
- reverse, forward, RANDOM ORDER, arbitrary mixed order in same training run;
- variable query hop count k in the SAME model (k mixed during training);
- repeated/ambiguous keys; 8/16/24 pairs; 2K+ contexts as next stage;
- plain tuned Transformer, original P-doubling, predecessor-only, hybrid
  two-head, possible prefix read-only encoder, explicit multi-pass control;
- exact final-answer accuracy by order, k, and key alias count;
- 3+ seeds, time, trainable params, FLOPs, activation memory and latency.

Minimum proof of concept: random 7/3 >=0.8 and forward 4/3 >=0.8
without losing reverse 7/3, and substantial advantage over a matched
two-head original+predecessor combination with the SAME compute budget.
If it only reproduces reverse-order results, classify MECHANISM_REPLICATION,
not NEW WUKE PAPER. Text LM improvement is a separate test.

## Run old mechanism pilot (NOT a new algorithm)

  python3 -m unittest discover -s pilot -p 'test_*.py'
  bash pilot/run_3h.sh --device cuda

Result: .pilot_workspace_pointer_pointer/reports/decision.json
Source is cloned in ignored workspace only. The existing run may fail
random-order acceptance, consistently with the published source findings.
