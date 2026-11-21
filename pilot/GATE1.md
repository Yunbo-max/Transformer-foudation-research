# SG-Delta Stage-Gate 1 — Mechanism and Actual New-Fact Learning

## Scope (2026-11-21)
This gate tests only one-state SG-Delta with historical Query-conditioned
**both** full directional retention and protected Write. It does not promote the
older write-only version as the final method. Predicted-Skip remains a separate
branch and is NOT being executed in this gate.

## Current status
- Code and generated MQAR panels: COMMITTED
- Controlled 2D algebra test: RUN LOCALLY in CPU Python, with separate constant
  historical Key statistic and skewed historical Query statistic.
- Full GitHub checkout pytest / GPU pilot: NOT VERIFIED in target machine.
- Real-model retention benefit and LM scaling: NOT DEMONSTRATED.
- NO GPU results were found uploaded to this branch's runs/ directory.
  Note that runs/ is ignored; this does not prove no one ran it elsewhere.

## Scientific preconditions
1. Input facts have equal initial Key-write exposure; old Query accesses are
   skewed, with guaranteed hot and cold post-interference probes.
2. **All token positions create K/V in an autoregressive model.**
   Therefore repeated Query tokens also affect the learned Key/write distribution.
   The end-to-end synthetic test is NOT an intervention that holds all Key
   statistics fixed. Independent mathematical test holds the Key statistic
   fixed explicitly and tests the distinct effect of Query statistics.
3. After distractor insertions, evaluate genuinely NEW Key->Value associations.
   Old fact retention without new fact acquisition is a failed hypothesis,
   not a positive result.
4. Reuse exactly the same paired data generator, training steps, decoder,
   optimizer, teacher-forcing masks and initial seeds for each method.
5. Retention conditional on learning: inspect both old-post accuracy and
   new-post accuracy, not only average answer NLL.

## Execute, once hardware is available

Use existing Python 3.10+ PyTorch CUDA environment; no data download needed.
From repo root, on this exact branch:

    python3 -m unittest discover -s pilot -p 'test_*.py'
    python3 pilot/mechanism_check.py > mechanism_evidence.json
    python3 pilot/run.py --profile smoke --methods isotropic query-joint --seeds 0 --device cuda
    bash pilot/run_3h.sh --profile pilot --device cuda

The shell launcher enforces 10800 s total on Linux and sets a 10200 s
internal experiment budget. On RTX 4090 24 GB use the pilot defaults,
NOT sizing-only 93.9M/308.6M configs. CPU host RAM 16-32 GB, free SSD >=5 GB.

## Arms and metrics

| Arm | Write | Decay | Role |
| --- | --- | --- | --- |
| isotropic | unchanged key | scalar | unprotected baseline |
| query-write | full Query geometry | scalar | old SG-Delta ablation |
| query-decay | isotropic | full Query-aware | decay ablation |
| query-joint | full Query geometry | full Query-aware | SOLE NEW PRIMARY |
| key-joint | full Key geometry | full Key-aware | geometric control |
| channel-decay | isotropic | learned diagonal | complexity control |
| reverse-joint | query-direction opposite retention | full | negative control, optional |

Primary evaluation: old_pre, old_post, old_post_hot, old_post_cold,
new_post (genuinely new facts), alias (new key with old value). Report
masked answer NLL and accuracy for each, actual model size, train time and
peak VRAM. Every method sees the same held-out evaluation generator seed.

## Predeclared decision rule

If both complete paired seeds produce >=15% old-post AND >=15% genuinely
new-post accuracy, look for query-joint advantage >=3pp vs key-joint,
>=2pp vs write-only AND decay-only; and both new-post and alias no worse
than 3pp vs controls. If not learned at all, label INCONCLUSIVE_NOT_LEARNED.
If learned but not improved, NO_CLEAR_ADVANTAGE. With all improvements,
GO_TO_REPLICATION (not paper-ready).

Critical mathematical warning: the controlled fixed-Key numerical example
shows Query-joint can protect hot readouts relative to Key-joint while
Query-write-only has slightly lower hot readout MSE than Query-joint.
Thus the incremental benefit of Query-aware Decay is **unproven**. The
pilot should be allowed to fail; do not present algebra alone as evidence.

## After this gate
Only then consider single GPU training ~93.9M, real document-disjoint text,
and compare with official GDN-2, PGDN, OSDN and Memory by Design. The
non-diagonal D_q is NOT natively supported by NVIDIA's GDN-2 channel-wise
decay kernel, and requires a kernel-equivalence/performance check first.
