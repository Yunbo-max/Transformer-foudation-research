# Predicted-Skip: mathematical and implementation audit

Status: the corrections and checks in this revision are **generated, not
executed**. Source inspection and algebra do not establish CUDA parity,
benchmark performance or a benefit from skipping. The original cosine gate
is retained, including its known failure cases.

## Source and attribution

- Vuk Rosic, author implementation pinned at
  [`d0dc052749a97061f6daa6907371bd0a2ec33405`](https://github.com/vukrosic/llm-research-paper-drafts/blob/d0dc052749a97061f6daa6907371bd0a2ec33405/03-linear-attention-skip-lookups/mix.py).
  `--lpsE 1` already scales both write and erase in the chunked delta family.
  Per-head learned skip strengths, the previous-query/chunk-start predictor,
  shuffled-value control and causal Q/K/V short convolution also appear there.
  Joint suppression is therefore an inherited control, not a new invention of
  this repository. The present wrapper studies channel-wise GDN-2 integration.
- NVIDIA, [Gated DeltaNet-2](https://arxiv.org/abs/2605.22791), with the operator
  pinned at
  [`de8476de0fecae08fc16f0bf3154900244c61c73`](https://github.com/NVlabs/GatedDeltaNet-2/blob/de8476de0fecae08fc16f0bf3154900244c61c73/lit_gpt/gdn2_ops/chunk_gdn2.py).
  The recurrence, default beta guard, normalization boundary and guard's
  straight-through backward were inspected. NVIDIA code is not vendored here.
- Flash Linear Attention's `l2norm_fwd` uses
  `x / sqrt(sum(x**2) + 1e-6)`, with float32 reduction and output-dtype rounding.
  The local reference uses this convention; CUDA parity is still required for
  the actual installed dependency versions.

See `SOURCES.md` for the wider paper/code comparison.

## 1. The raw recurrence and its exact joint identity

Write the state in key-by-value orientation, S in R^(K x V):

    prior = D S_previous,                 D = diag(exp(g))
    z     = w * v - prior.T @ (b * k)
    S_raw = prior + outer(k, z).

Here `*` is elementwise multiplication, b acts on the key axis and w on the
value axis. The reference implements this order: decay, erase, write, read.
For nonzero k, the active edit is zero exactly when z = 0.

With raw gates and a scalar rho per token/head,

    b_skip = rho b,   w_skip = rho w
    S_joint = prior + rho (S_raw - prior).

The identity holds for a fixed incoming state even if rho was calculated
from that state. It does not imply that two entire trajectories are scalar
multiples of each other. Finite-precision rounding adds numerical error.

A useful limiting case shows why joint and write-only differ. Suppose D=I,
k is unit length, b and w equal a scalar beta, and S.T @ k = v. The baseline
and joint active edit are zero. Write-only instead changes the state by
`-beta * (1-rho) * outer(k,v)`, damaging an already correct association.
This is an algebraic observation, not evidence of a general accuracy gain.

## 2. What the predictor actually measures

For a chunk beginning at c, the gate is

    read_t = S_c.T @ q_(t-1)
    rho_t  = exp(-gamma_h * relu(cosine(read_t, v_t))).

S_c is the state before the current chunk. It does not include earlier
updates within that chunk. This is a causal approximation to a previous
read, not the exact S_(t-1).T @ q_(t-1). The first chunk has S_c=0 and rho=1.

The gate is a soft attenuation, not a conditional compute-skipping branch.
For positive finite gamma, rho is positive; even perfect cosine gives
`exp(-gamma)`, not zero. Decay is never suppressed by this gate.

High cosine does not establish z=0:

1. A new key may need to be associated with an already familiar value.
   The previous read can equal v while the new association is absent.
2. A weak read `epsilon * v` has the same cosine as a strong read, above the
   numerical epsilon region. A useful strengthening write can be attenuated.
3. The test does not include current k, b, w or the current decayed state.
   With independent channel-wise gates, even equality of an ungated read
   and v does not generally imply cancellation of erase and write.

`oracle_diagnostic.py` and the algebraic tests preserve these counterexamples.
They are diagnostic examples, not replacements for native benchmark scoring.
No residual veto or extra key-conditioned predictor is introduced here.

### Decay plus suppressed refresh

For repeated unit key k, scalar b=w=beta, scalar decay alpha<1 and a positive
same-direction read, cosine remains one as its magnitude changes. If
`m_t = S_t.T @ k` and rho is consequently constant,

    m_t = alpha * (1-rho*beta) * m_(t-1) + rho*beta*v
    m_infinity = rho*beta / (1-alpha+alpha*rho*beta) * v.

With alpha=.99, beta=.5 and rho=exp(-5), the fixed-point amplitude is about
.253 of v, versus about .990 with rho=1. This is an analytic example of why
suppressing active edits while retaining decay does not guarantee memory
preservation. The original predictor remains available so this limitation
can be measured rather than hidden by an unreviewed mechanism change.

## 3. Guard semantics are explicit

`guard_mode='raw'` uses the unprojected recurrence and supports the exact
identity above. `guard_mode='native'` applies the GDN-2 beta guard **after**
skip scaling, matching the placement in NVIDIA's operator. Native is the
new default; raw must be requested for the historical mathematical control.
The legacy `beta_guard` boolean maps to these policies; contradictory
arguments raise an error. Reference and official backends implement the
same requested policy, rather than silently ignoring the reference flag.

### Independent reference implementation of the certificate

For normalized k, define

    n  = sum(k_i^2)
    a  = sum(b_i * k_i^2)
    nu = sum(b_i^2 * k_i^2)
    r2 = max(n*nu-a^2, 0)
    c2 = exp(-2*max_i(g_i))
    limit = max((c2-1) * (1-(1-a)^2/c2), 0).

The guard leaves a row unchanged when its contrast r2 is within limit.
Otherwise it contracts b about the key-weighted mean a/n, retaining a
31/32 margin inside the bound. It rounds to b's dtype, rechecks the
certificate using the upstream rounding tolerance, and falls back to the
rounded scalar mean if needed. Zero-key rows remain unchanged.

The implementation keeps the effective projected value in the forward and
uses an identity derivative with respect to b. The projection supplies no
extra derivatives through k or g, matching NVIDIA's straight-through policy.
This is an independent tensor implementation of the mathematical operation;
its numeric agreement with the CUDA version must pass the supplied checks.
The CUDA guard recovers per-token g from a chunk-local cumulative log;
rounding near a guard boundary is covered by the parity tests.

Projection P is not generally homogeneous: P(rho*b) need not equal
rho*P(b). Consequently **native-guard joint suppression is not claimed to
satisfy the raw scalar-edit identity relative to native baseline**. Both
policies must remain identifiable in results. No extra guard-before-joint
variant is added in this revision.

Normalized k and sigmoid b alone do not establish Euclidean non-expansion:
`I - outer(k,b*k)` is generally non-symmetric. The guard addresses a
conditional linear state-transition certificate. A state-dependent skip
coefficient also has a derivative with respect to the state; the certificate
alone is not a proof of global contraction of that nonlinear mapping.

## 4. Gradient and representation conventions

- The predictor's memory input is detached when `gate_memory_detach=True`.
  This removes the gate's derivative through S_c only. The main recurrent
  state path retains gradients across chunks. This is an explicit surrogate
  gradient policy, not full differentiation of all forward dependencies.
  The CLI exposes its alternative and the parity receipt records the choice.
  Vuk's pinned chunked implementation differentiates through its boundary
  state. Our default is therefore a training-policy difference, not an exact
  reproduction of that backward path or an established improvement over it.
- Both backends take raw Q/K at their operator boundary and normalize once.
  Gate probes use the same normalized raw projections. The old adapter fed
  already normalized Q/K to an operator that normalized them again.
- The reference accumulates the recurrent state in float32, including under
  autocast. Outputs are cast to the projection dtype. The decay activation
  is calculated in float32; the final state is float32 on both backends.
- `conv_size=4` supplies identity-initialized causal depthwise Q/K/V
  convolution followed by SiLU, as motivated by the inspected author
  architecture. All experimental arms share it. `conv_size=0` retains the
  old pilot's linear Q/K and SiLU value path. The wrapper still is not a full
  reproduction of NVIDIA's decoder architecture.
- Gamma is learned separately per head and initialized to .25; `gamma=2`
  remains selectable. The per-head parameter has a different shape and
  training constraint from the former single global scalar. Changing these
  settings is a new experimental configuration, not an exact replay of old
  checkpoints/results. Gamma is capped at 20 before exponentiation to avoid
  overflow in the old exp-then-clamp parameterization.

## 5. Diagnostics do not change the gate

`diagnostics=True` collects per-token rho, cosine, read norm and value norm
during evaluation. The reference backend also reports the baseline and
applied active-residual/edit norms plus guard adjustment. The baseline
residual here means the unskipped edit evaluated at the **actual current
trajectory's prior state**, with the selected guard policy. It is not a
counterfactual baseline run. Official diagnostics do not fabricate residuals
that the unchanged kernel does not expose.

With --diagnostics, common.py saves these measurements under gate_events,
separately by layer and event: initial-fact values, alias-fact values, new-fact
values and the answer groups. Each metric retains its sum and count. Panel
means divide pooled sums by pooled head-token counts. Group totals are not
summed together because old_post overlaps its hot/cold subgroups.

The scorer evaluates an answer at token position t using the logit at t-1,
whereas its answer-value write occurs at t. Event masks explicitly shift the
loss-group masks right by one token. Fact masks mark the value in [FACT,k,v].
These labels are evaluator metadata, never predictor inputs. The current
alias query immediately follows its fact; there is no delayed-alias task.

This can reveal which token events receive attenuation and whether their
unskipped active edits are small. It is not a skip-correctness label: a large
edit can still be harmful, and causal convolution may carry necessary binding
information beyond the fact-value token. Learning outcomes remain necessary.

## 6. What validation establishes

`test_methods.py` contains targeted algebra, guard, causality, normalization
boundary and gradient-connectivity checks. The alias and refresh examples
intentionally demonstrate limitations. Their presence is not a performance
claim, and the generated tests have not been executed in the Web host.
`test_diagnostics.py` adds hand-enumerated event-time checks, unequal-count
aggregation checks and checks that absent diagnostics are not fabricated.

`validate_official.py` checks raw/native policies against the pinned clean
NVIDIA checkout. It covers one-token and partial chunks and the CUDA
operator's internal 64-token boundary. It compares output, final state, all
primitive q/k/v/g/b/w/initial-state gradients, model input gradients and all
used model parameter gradients. A baseline's unused gamma is reported
explicitly; missing gradients in other used parameters fail the check.

The receipt records the core source hash, official commit, guard policies,
methods, exact model configuration, runtime and precision. Its separate
`verification_sha256` binds both `validate_official.py` and the shared
`parity_contract.py`, so changing validation logic or tolerances invalidates
an earlier receipt. Training compares the full shared runtime fingerprint:
Python, Torch/CUDA/cuDNN, GPU name and capability, Triton and Flash Linear
Attention module/distribution versions, the SHA-256 of the source file
defining the actually imported FLA `l2norm_fwd`, and matmul/TF32/determinism
flags. Paths are not compared; code identity is determined from source bytes.
The validator also checks that the fingerprint did not change while it ran.
Old receipts lacking these fields fail closed. Dependency imports and GPU
inspection occur only when the official runtime fingerprint is requested;
the reference path does not require FLA or Triton. Float32 and
bfloat16 receive separate receipts; a BF16 test cannot authorize a FP32
training configuration. Both elementwise and relative-L2 criteria must pass.
The tolerances are prespecified per dtype in the validator and results are
saved even when checks fail. A receipt validates implementation parity for
its stated configuration only, not the scientific usefulness of Predicted-Skip.
