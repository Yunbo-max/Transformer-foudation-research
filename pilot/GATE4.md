# Wuke Candidate 4 — Adaptive Muon Stage Gate (2026-11-22)

## What already exists
Original coauthor: https://github.com/vukrosic/llm-research-paper-drafts/tree/main/02-muon-hypergradient-learning-rate
Pinned MIT source: d0dc052749a97061f6daa6907371bd0a2ec33405

Muon orthogonalizes the momentum update O. The original paper learns ONE
global rate multiplier s using a hypergradient cosine between CURRENT raw
gradient G_t and PREVIOUS Muon update direction O_{t-1}:
  c_t = mean_matrices cos(G_t, O_{t-1})
  log s_(t+1) = log s_t + gain * c_t
  actual rate = base_schedule(t) * s_t
The original work already contains a fixed early adaptation window,
hold/freeze, asymmetric gain, noise-relative post-hold downward corrections,
and width scaling. It reports 42/45 small-setting results within 0.005
validation NLL of a tuned sweep; this result is NOT ours.

## Current Wuke candidate — confidence-triggered freeze
The EXISTING pilot patch adds --dh_patience 35 to ss.py, and stops adaptation
after 35 consecutive steps whose global EMA cosine magnitude is below
0.6 times the running cosine standard deviation; fixed 30%-window hold and
noise-sensitive post-hold lower-only rules remain comparators.

This is only a small hypothesis and NOT novel by itself:
online stepsize adaptation, noisy signal detection, freeze/schedule policies,
and Muon magnitude control have major prior work.
Additional important competitors:
* Chu et al., ICML 2025: Provable and Practical Online Learning Rate Adaptation
  with Hypergradient Descent, https://proceedings.mlr.press/v267/chu25a.html
* Crawshaw et al., ICML 2026: MuonMax + Momo non-Euclidean adaptation,
  https://proceedings.mlr.press/v306/crawshaw26a.html
* Yuan 2026: OptMuon closed-loop magnitude,
  https://arxiv.org/abs/2606.08783
* Glentis et al. 2026: MuonGS weight-norm dynamics,
  https://arxiv.org/abs/2610.37647

## Counterexample-first experimental design
The original paper identifies three failures worth testing separately:
(1) tenfold high initial LR needs more than the fixed hold window to recover;
(2) very short runs can learn the wrong rate for their horizon;
(3) late biased noisy alignment makes unbounded adaptation shrink LR.

Current pilot (code already committed in run.py):
 - width64 / 4 layers / 225 steps / batch16 (short-run failure)
 - width128 / 4 layers / 900 steps / batch4 (slow recovery from high LR)
 - context64, vocabulary top-1000, same training windows, seeds 0/1
 - 3 fixed Muon LR controls (0.00125, 0.005, 0.02)
 - original 30%-hold hypergradient, original 2-sigma post-hold correction,
   and Wuke confidence patience=35
 - target single RTX 4090 24GB, host RAM >=16GB, <=3h shell budget
 - synthetic Zipf/topic by default: mechanism PROXY, not FineWeb evidence
 - optional real train/val shards via --real-data-dir, genuinely disjoint

Log/compare: held-out NTP NLL, loss spikes, divergence, LR multiplier trace,
adaptation/freeze time (must add explicit freeze-step logging before a
publication claim), run seconds. The PRESENT runner tests a high initial rate of 0.05 and a
three-rate fixed sweep; a separate 10x-low initial-rate test is PLANNED,
not yet wired into the runner.

Do not claim a GO from a single easy model: require both hard configuration
seeds to improve 30%-hold by 0.005 nats and lie within 0.006 of the best
three-rate fixed baseline; additionally check confidence method against the 2-sigma post-hold baseline (>=0.002 nats
better is used as a conservative screen in the current runner).
Synthetic-only wins remain INCONCLUSIVE.

## Main research decision
The published adaptive optimizer literature is CROWDED. Confidence-hold
alone is unlikely to be a sufficiently independent paper. A stronger
hypothesis must solve the exact failure: decide WHEN evidence supports
freezing without stopping recovery from an initially excessive rate.
A possible *future* test uses BOTH confidence of gradient-vs-previous-update
alignment and stabilization of the LR trajectory, with explicit fallback
when the alignment is uninformative. Do not invent a novelty claim or
code a combined controller until its mechanism is justified by the
existing baseline comparisons.

If this screen only ties, stop the optimizer branch and move to Wuke
Candidate 5, Identity + Pointer Doubling, rather than adding optimizer
heuristics indefinitely.

## Reproduction
On branch pilot/adaptive-muon, from repository root:
  python3 -m unittest discover -s pilot -p 'test_*.py'
  bash pilot/run_3h.sh --device cuda
For actual document-disjoint GPT-2 uint16 text:
  bash pilot/run_3h.sh --device cuda --real-data-dir /data/fineweb_bins
Outputs: .pilot_workspace_muon/reports/{report.json,decision.json,*.log}
No GPU results have been verified in this stage.
