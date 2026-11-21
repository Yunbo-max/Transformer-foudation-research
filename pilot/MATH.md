# SG-Delta: exact objective, implementation contract, and source review

Status of this revision: **source-reviewed; revised mathematical tests and
runtime behavior await local execution**. No GPU, real-text, official-baseline,
or scientific-performance result is established by this document.

The approved main method remains joint historical-query directional retention
and protected writing. The recurrent equations, scalar fast/slow gate
parameterization, and constructor defaults are preserved. Numerical precision
handling and independently checkable reference functions were added.
The experiment runner may explicitly choose a different convolution setting;
that setting must be recorded and matched across arms.

## 1. State convention and causal geometry

For one head, the content memory is $S_t\in\mathbb R^{K\times V}$,
$q_t,k_t\in\mathbb R^K$, and $v_t\in\mathbb R^V$. Reading uses
$S_t^\top q_t$. Covariance $C_c\in\mathbb R^{K\times K}$ is an auxiliary
recurrent statistic, not an additional content-memory bank.

At the beginning of chunk $c$, $C_c$ includes only completed earlier chunks.
For a completed chunk of length $L$,

$$
C_{c+1}
=\rho^L C_c
 +(1-\rho)\sum_{j=0}^{L-1}\rho^{L-1-j}q_{c,j}q_{c,j}^{\top}.
$$

This equals a tokenwise EMA. Starting from zero, $C_c\succeq0$ in exact
arithmetic. The model normalizes projected queries, so
$\operatorname{tr}C_c\le1$. If all observed query norms equal one and there
have been $n$ observations, the trace is $1-\rho^n$.

All token queries contribute, including ordinary text tokens. This statistic
measures historical query usage; it does not establish semantic importance,
successful retrieval, or which factual association should be retained.
No gold answer mask is supplied to the mixer.

The default stops gradients through $C$. This changes the training
derivative, not the forward recurrence. Differentiable statistics are an
explicit alternative and must be identified in run metadata.

## 2. Joint recurrence

Let $\lambda>0,\tau>0,\mu_{\mathrm{eff}}\ge0$, and
$0<\alpha_f\le\alpha_s\le1$. Define

$$
H=\lambda I+\mu_{\mathrm{eff}}C,\qquad
P=I-\lambda H^{-1},\qquad
D=\alpha_f I+(\alpha_s-\alpha_f)P.
$$

Then

$$
\begin{aligned}
\bar S&=DS^-,\\
e&=v-\bar S^\top k,\\
a&=\frac{H^{-1}k}{\tau+k^\top H^{-1}k},\\
S^+&=\bar S+ae^\top,
\end{aligned}
$$

where $S^-=S_{t-1}$. A Cholesky factorization is reused for every token in
the chunk. The residual reads with the original $k$; only the write
direction is preconditioned. The residual is evaluated after directional
decay. Changing either of these operations changes the mathematical method.

## 3. A single exact objective for decay and write

The update is the unique minimizer of

$$
\boxed{
J(S)
=
\frac12\|S^\top k-v\|_2^2
+\frac{\tau\lambda}{2}\|S-\alpha_f S^-\|_F^2
+\frac{\tau\mu_{\mathrm{eff}}}{2}
 \operatorname{tr}\!\left[
 (S-\alpha_s S^-)^\top C(S-\alpha_s S^-)
 \right].
}
$$

The Hessian for each value channel is
$kk^\top+\tau(\lambda I+\mu_{\mathrm{eff}}C)\succ0$, so the minimizer is
well-defined even when $C$ is rank-deficient or $k=0$.

The first-order condition is

$$
(kk^\top+\tau H)S
=
kv^\top+
\tau(\lambda\alpha_f I+\mu_{\mathrm{eff}}\alpha_s C)S^-.
$$

Because $H=\lambda I+\mu_{\mathrm{eff}}C$,

$$
\begin{aligned}
H^{-1}(\lambda\alpha_f I+\mu_{\mathrm{eff}}\alpha_s C)
 &=\alpha_s I-\lambda(\alpha_s-\alpha_f)H^{-1}\\
 &=D.
\end{aligned}
$$

The two regularizers therefore combine, up to an $S$-independent constant,
into $\frac{\tau}{2}\operatorname{tr}[(S-\bar S)^\top H(S-\bar S)]$.
Solving this quadratic with Sherman--Morrison gives the implemented $a$
and $e$.

If $C=\sum_i w_iq_iq_i^\top$, the query regularizer is

$$
\sum_i w_i
\left\|S^\top q_i-\alpha_s(S^-)^\top q_i\right\|_2^2.
$$

Thus it protects prior readouts relative to slow retention, while the
isotropic term pulls the whole state toward fast retention. It is a soft
tradeoff. Conflicting old and new targets for the same direction cannot
both be represented exactly in one linear memory.

This derivation applies to the joint arm, including the key-statistic
comparator after replacing $C_q$ by $C_k$. Component ablations intentionally
remove one part of the joint construction.

## 4. Spectral properties and the precise stability statement

For $C=U\operatorname{diag}(c_i)U^\top$, $c_i\ge0$,

$$
p_i=\frac{\mu_{\mathrm{eff}}c_i}{\lambda+\mu_{\mathrm{eff}}c_i},
\qquad
d_i=\frac{\lambda\alpha_f+\mu_{\mathrm{eff}}c_i\alpha_s}
           {\lambda+\mu_{\mathrm{eff}}c_i}.
$$

Hence $0\le p_i<1$ and $\alpha_f\le d_i\le\alpha_s$. Increasing a
covariance eigenvalue slows decay along its eigenvector. The same geometry
also decreases the corresponding unnormalized write component through
$H^{-1}$.

For $r=k^\top H^{-1}k$, the current-key residual obeys

$$
v-(S^+)^\top k
=\frac{\tau}{\tau+r}\big(v-\bar S^\top k\big).
$$

This is contraction of the current regression residual relative to the
decayed state; it is not a guarantee that every historical readout improves.

For two states receiving identical current inputs and fixed geometry, their
difference evolves through

$$
M=(I-ak^\top)D.
$$

Using $\|X\|_H^2=\operatorname{tr}(X^\top HX)$ and
$w=H^{-1/2}k$,

$$
H^{1/2}(I-ak^\top)H^{-1/2}
=I-\frac{ww^\top}{\tau+\|w\|^2}.
$$

This transformed write operator is symmetric with eigenvalues in $(0,1]$.
Since $D$ commutes with $H$, it follows that

$$
\|MX\|_H\le\alpha_s\|X\|_H.
$$

This statement uses **fixed $H$**, which holds inside a chunk. It does not
prove Euclidean contraction, a bound on the affine input $av^\top$, or
global stability across changing chunk metrics. Those require separate
arguments and runtime measurements.

When $\mu_{\mathrm{eff}}=0$ or $C=0$, the joint rule reduces to

$$
S^+=\alpha_fS^-+
\frac{k}{\tau\lambda+\|k\|^2}
\left(v-\alpha_f(S^-)^\top k\right)^\top.
$$

For unit keys and default $\tau=\lambda=1$, this has an effective write
coefficient of $1/2$. It is a particular isotropic baseline, not a faithful
replacement for every learned-gate GDN or official GDN-2 configuration.

## 5. Geometry scale is explicit

`geometry_scale="raw"` is the constructor default and preserves the original
interpretation $\mu_{\mathrm{eff}}=\mu$. Since the normalized EMA has trace
at most one, approximately isotropic queries yield $C\approx I/K$, so

$$
P\approx\frac{\mu}{\lambda K+\mu}I.
$$

At $\lambda=1,\mu=8$, this fraction is $1/3$ for $K=16$ and $1/9$
for $K=64$. Keeping the same numeric $\mu$ across head dimensions does
not keep the same average protection strength.

`geometry_scale="dimension"` is an explicit opt-in parameterization,
$\mu_{\mathrm{eff}}=K\mu$. In the same isotropic limit, the fraction becomes
$\mu/(\lambda+\mu)$. It must not be silently enabled or presented as
unchanged hyperparameters. The runner records the chosen scale, base $\mu$,
and architecture; the mixer also returns `effective_mu` and
`geometry_scale_factor`.

There is no automatic trace normalization, clipping, or retuning.

## 6. Numerical and feature-mixing contract

The pure functions used by the actual mixer are:

- `query_geometry`: construct and factor the full SPD precision.
- `protected_write_directions`: solve all chunk write directions once.
- `sg_delta_step`: decay, residual read, and rank-one write.
- `update_query_covariance`: consume only a completed chunk.

These operations accumulate in FP32; explicit FP64 inputs preserve FP64.
Autocast is disabled around normalization, covariance, Cholesky, recurrence,
and readout. Projection/convolution layers and the final output projection
may use the surrounding mixed-precision context. Casting projections into
FP32 does not undo quantization already incurred in those projections.

The covariance/protection matrices are symmetrized to remove rounding
asymmetry. There is no hidden diagonal jitter, eigenvalue clipping, or retry
that changes $\lambda$. A failed Cholesky remains an observable error.

`conv_size=0` preserves the original constructor feature path. A positive
size enables independent projected q/k/v depthwise convolutions, followed
by SiLU and then q/k normalization. Padding is exclusively on the left.
This optional wrapper provides short-range context before recurrent memory;
it does not alter the SG joint state-update equations. It changes parameters
and features, so compare arms under the same convolution setting and do not
equate runs across settings.

The current implementation resets $S$ and $C$ for each full-sequence
forward call and returns no persistent decoding cache. It is not yet a
streaming generation implementation. State-memory accounting includes both
the content memory and the historical covariance; actual allocated peak
memory must be measured independently.

`query_geometry_fraction` is the mean diagonal of $P$.
`importance_fraction` remains a compatibility alias, not a claim of semantic
importance. `actual_mean_decay` reports the decay actually applied; the
fast/slow gates alone do not reveal that average.

## 7. Primary implementation sources inspected

These sources informed numerical handling and the optional feature wrapper.
Their code is not copied into this project, and their kernels do not provide
SG-Delta's full non-diagonal query-conditioned decay.

1. **PGDN**, pinned `7bd753279af87b39114149a104c5bde9bf67145f`:
   [reference recurrence](https://github.com/ntumm120/preconditioned-deltanet/blob/7bd753279af87b39114149a104c5bde9bf67145f/3rdparty/flash-linear-attention/fla/ops/precond_gated_delta_rule/naive.py)
   explicitly converts recurrence inputs to FP32, reads the residual with
   the original key, and writes with a preconditioned key. Its statistics
   are diagonal key statistics, not this full query covariance.
   [layer implementation](https://github.com/ntumm120/preconditioned-deltanet/blob/7bd753279af87b39114149a104c5bde9bf67145f/3rdparty/flash-linear-attention/fla/layers/precond_gated_deltanet.py)
   uses projected q/k/v short convolutions and SiLU, with a default
   convolution size of four.
   [paper](https://arxiv.org/abs/2604.21100).

2. **OSDN**, pinned `39406acc683a562f8a4aa303daaffd3ddcec4db8`:
   [reference recurrences](https://github.com/Lhongpei/OSDN/blob/39406acc683a562f8a4aa303daaffd3ddcec4db8/src/osdn/reference.py)
   keep state and preconditioner accumulation in FP32, expose pure reference
   functions, preserve the original-key residual read, and explicitly store
   the pre-update preconditioner trajectory.
   [reference tests](https://github.com/Lhongpei/OSDN/blob/39406acc683a562f8a4aa303daaffd3ddcec4db8/tests/test_reference.py)
   check baseline limits and differentiability.

3. **SG-Delta input revision**:
   `edf50b3e961823e6110fa4472590ed1573f957a2` on
   `pilot/sgdelta-joint-query-decay-write`. This repair preserves its approved
   joint recurrence. It does not constitute a reproduction of PGDN/OSDN.

## 8. Mathematical checks supplied for local execution

The revised test suite checks the actual pure functions against an independent
direct solve of the unified objective, including off-diagonal covariance,
non-unit and zero keys, and different $\lambda,\mu,\tau$. It also checks:

- Stationarity of the unified objective and the exact residual-shrink factor.
- Spectral retention on rotated covariance eigenspaces.
- The isotropic limit in both the pure functions and the full mixer.
- Chunk EMA equivalence to a tokenwise reference, including $\rho=0$.
- Same-chunk and cross-chunk causality, with convolution disabled/enabled.
- Fixed-geometry contraction in the $H$-norm.
- Explicit stop-gradient behavior.
- FP16/BF16 input promotion, FP64 preservation, and autocast isolation.
- Explicit geometry scaling and failure on invalid parameters.
- End-to-end gradients with statistics detached and differentiable.

These are numerical/algebraic regression checks. Passing them would validate
their tested properties, not scientific superiority or a paper-scale result.
The tests have been authored but not executed in this source-review session.

Local command:

~~~bash
python3 -m unittest discover -s pilot -p 'test_*.py'
~~~
