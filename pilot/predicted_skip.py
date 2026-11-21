"""Predicted-Skip controls around a small GDN-2 reference decoder.

The cosine predictor, chunk-start read and joint erase/write option already
exist in Vuk Rosic's author implementation (mix.py --lpsE 1). This module
studies their integration with channel-wise GDN-2 gates; it does not claim the
joint option as a new invention. See MATH.md for source pins and limitations.

raw uses the unprojected recurrence. native applies the GDN-2 beta-contrast
guard AFTER skip scaling. The independently written PyTorch counterpart uses
the same straight-through gradient convention as the official operator.
"""
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ARMS = ("baseline", "write", "joint", "shuffled", "keyread")
GUARD_MODES = ("raw", "native")
QK_EPS = 1e-6
MAX_GAMMA = 20.0


def normalize_qk(x):
    """One FLA-style normalization: sqrt(sum(x**2) + eps), in float32."""
    xf = x.float()
    return (xf * torch.rsqrt(xf.square().sum(-1, keepdim=True) + QK_EPS)).to(x.dtype)


def beta_guard(k, b, g):
    """Independently implement the published beta-contrast certificate.

    Inputs have equal shape [..., K]; k is normalized and g is per-token log
    decay. Include output-dtype rounding and the certificate recheck. Only the
    identity derivative with respect to b is retained, matching the official
    straight-through backward; the projection adds no k/g derivatives.
    """
    if k.shape != b.shape or b.shape != g.shape or k.ndim < 1:
        raise ValueError("guard k, b and g must have identical [..., K] shapes")
    if not all(t.is_floating_point() for t in (k, b, g)):
        raise TypeError("guard inputs must be floating-point tensors")
    if not (k.device == b.device == g.device):
        raise ValueError("guard inputs must share a device")
    with torch.no_grad(), torch.autocast(device_type=k.device.type, enabled=False):
        key_energy = k.float().square()
        total_energy = key_energy.sum(-1, keepdim=True)
        safe_energy = total_energy.clamp_min(1e-20)
        original = b.float()
        radius_squared = torch.exp(-2 * g.float().amax(-1, keepdim=True))

        def certificate(candidate):
            moment = (candidate * key_energy).sum(-1, keepdim=True)
            second = (candidate.square() * key_energy).sum(-1, keepdim=True)
            contrast = (total_energy * second - moment.square()).clamp_min(0)
            allowed = ((radius_squared - 1) *
                       (1 - (1 - moment).square() / radius_squared)).clamp_min(0)
            return moment, second, contrast, allowed

        moment, _, contrast, allowed = certificate(original)
        needs_projection = (total_energy > 1e-20) & (contrast > allowed)
        center = moment / safe_energy
        contraction = torch.sqrt((31 / 32) * allowed / contrast.clamp_min(1e-30)).clamp(0, 1)
        projected = center + contraction * (original - center)
        rounded = torch.where(needs_projection, projected, original).to(b.dtype).float()
        rounded_moment, rounded_second, rounded_contrast, rounded_allowed = certificate(rounded)
        rounding_tolerance = 4 * torch.finfo(b.dtype).eps * (
            total_energy * rounded_second + rounded_moment.square())
        needs_scalar = needs_projection & (rounded_contrast > rounded_allowed + rounding_tolerance)
        effective = torch.where(needs_scalar, rounded_moment / safe_energy, rounded).to(b.dtype)
    # This forward is exactly effective even in low precision, with db = I.
    return effective.detach() + (b - b.detach())


class CausalDepthwiseConv(nn.Module):
    """Left-padded depthwise convolution, identity-initialized, then SiLU."""
    def __init__(self, channels, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(channels, channels, kernel_size, groups=channels, bias=False)
        with torch.no_grad():
            self.conv.weight.zero_()
            self.conv.weight[..., -1] = 1.0

    def forward(self, x):
        z = F.pad(x.transpose(1, 2), (self.kernel_size - 1, 0))
        return F.silu(self.conv(z).transpose(1, 2))


class GDN2PredSkipMixer(nn.Module):
    def __init__(self, hidden, heads=2, key_dim=16, value_dim=None, chunk=16,
                 mode="joint", gamma=.25, backend="reference", official_root=None,
                 beta_guard=None, gate_memory_detach=True, *, guard_mode=None,
                 conv_size=4, diagnostics=False):
        super().__init__()
        value_dim = key_dim if value_dim is None else value_dim
        for name, value in (("hidden", hidden), ("heads", heads), ("key_dim", key_dim),
                            ("value_dim", value_dim), ("chunk", chunk)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(conv_size, bool) or not isinstance(conv_size, int) or conv_size < 0:
            raise ValueError("conv_size must be a nonnegative integer")
        if mode not in ARMS or backend not in ("reference", "official"):
            raise ValueError("unknown method or backend")
        if not math.isfinite(gamma) or not 0 < gamma <= MAX_GAMMA:
            raise ValueError(f"gamma must lie in (0, {MAX_GAMMA}]")
        if beta_guard is not None and not isinstance(beta_guard, bool):
            raise TypeError("legacy beta_guard must be bool or None")
        legacy_guard = None if beta_guard is None else ("native" if beta_guard else "raw")
        if guard_mode is not None and legacy_guard is not None and guard_mode != legacy_guard:
            raise ValueError("guard_mode conflicts with legacy beta_guard")
        guard_mode = guard_mode or legacy_guard or "native"
        if guard_mode not in GUARD_MODES:
            raise ValueError(f"guard_mode must be one of {GUARD_MODES}")
        if backend == "official" and key_dim > 256:
            raise ValueError("official GDN-2 supports key_dim <= 256")
        if guard_mode == "native" and not 2 <= key_dim <= 256:
            raise ValueError("native beta guard requires 2 <= key_dim <= 256")
        self.hidden = hidden
        self.heads, self.kdim, self.vdim = heads, key_dim, value_dim
        self.chunk, self.mode, self.backend = chunk, mode, backend
        self.official_root = official_root
        self.guard_mode = guard_mode
        self.beta_guard = guard_mode == "native"
        self.gate_memory_detach = bool(gate_memory_detach)
        self.conv_size, self.diagnostics = conv_size, bool(diagnostics)
        self.q_proj = nn.Linear(hidden, heads * key_dim, bias=False)
        self.k_proj = nn.Linear(hidden, heads * key_dim, bias=False)
        self.v_proj = nn.Linear(hidden, heads * value_dim, bias=False)
        self.erase_proj = nn.Linear(hidden, heads * key_dim)
        self.write_proj = nn.Linear(hidden, heads * value_dim)
        self.decay_proj = nn.Linear(hidden, heads * key_dim)
        self.output = nn.Linear(heads * value_dim, hidden, bias=False)
        self.q_conv = CausalDepthwiseConv(heads * key_dim, conv_size) if conv_size else None
        self.k_conv = CausalDepthwiseConv(heads * key_dim, conv_size) if conv_size else None
        self.v_conv = CausalDepthwiseConv(heads * value_dim, conv_size) if conv_size else None
        self.log_gamma = nn.Parameter(torch.full((heads,), math.log(gamma)))
        self._official_kernel = None
        self.last_rho = None
        self.last_diagnostics = None

    def official_kernel(self):
        if self._official_kernel is None:
            if not self.official_root:
                raise RuntimeError("official backend requires --official-root")
            root = Path(self.official_root).expanduser().resolve()
            if not (root / "lit_gpt/gdn2_ops/chunk_gdn2.py").is_file():
                raise FileNotFoundError("expected NVIDIA GatedDeltaNet-2 checkout")
            sys.path.insert(0, str(root))
            from lit_gpt.gdn2_ops.chunk_gdn2 import chunk_gdn2
            self._official_kernel = chunk_gdn2
        return self._official_kernel

    def _check_chunk(self, q, k, v, g, b, w, state):
        if q.ndim != 4 or q.shape != k.shape or q.shape != g.shape or q.shape != b.shape:
            raise ValueError("q/k/g/b must have equal [batch, heads, time, key_dim] shapes")
        if q.shape[1] != self.heads or q.shape[-1] != self.kdim or q.shape[2] < 1:
            raise ValueError("chunk dimensions do not match this mixer")
        if v.shape != (*q.shape[:3], self.vdim) or w.shape != v.shape:
            raise ValueError("v/w must have matching value dimensions")
        if state.shape != (q.shape[0], self.heads, self.kdim, self.vdim):
            raise ValueError("initial state has an incorrect shape")

    def _reference_chunk(self, q, k, v, g, b, w, state, *, baseline_gates=None,
                         diagnostic_rows=None):
        """Raw Q/K inputs, normalized exactly once, with float32 recurrence."""
        self._check_chunk(q, k, v, g, b, w, state)
        qn, kn = normalize_qk(q), normalize_qk(k)
        effective_b = beta_guard(kn, b, g) if self.beta_guard else b
        original_b, original_w = baseline_gates if baseline_gates is not None else (b, w)
        if diagnostic_rows is not None and self.beta_guard:
            original_b = beta_guard(kn, original_b, g)
        ys = []
        with torch.autocast(device_type=q.device.type, enabled=False):
            state = state.float()
            for t in range(q.shape[2]):
                kt, vt, qt = kn[:, :, t].float(), v[:, :, t].float(), qn[:, :, t].float()
                prior = state * torch.exp(g[:, :, t].float()).unsqueeze(-1)
                erase = torch.einsum("bhk,bhkv->bhv", effective_b[:, :, t].float() * kt, prior)
                written = w[:, :, t].float() * vt
                residual = written - erase
                state = prior + kt.unsqueeze(-1) * residual.unsqueeze(-2)
                ys.append(torch.einsum("bhkv,bhk->bhv", state, qt) * self.kdim ** -.5)
                if diagnostic_rows is not None:
                    with torch.no_grad():
                        unskipped = original_w[:, :, t].float() * vt - torch.einsum(
                            "bhk,bhkv->bhv", original_b[:, :, t].float() * kt, prior)
                        key_norm = kt.norm(dim=-1)
                        local = {
                            "baseline_residual_norm": unskipped.norm(dim=-1),
                            "active_residual_norm": residual.norm(dim=-1),
                            "baseline_edit_norm": key_norm * unskipped.norm(dim=-1),
                            "applied_edit_norm": key_norm * residual.norm(dim=-1),
                            "guard_delta_norm": (effective_b[:, :, t].float() - b[:, :, t].float()).norm(dim=-1),
                        }
                        for name, value in local.items():
                            diagnostic_rows.setdefault(name, []).append(value.detach())
        return torch.stack(ys, 2).to(q.dtype), state

    def _official_chunk(self, q, k, v, g, b, w, state):
        """Pass raw Q/K; the official operator performs their ONLY normalization."""
        self._check_chunk(q, k, v, g, b, w, state)
        kernel = self.official_kernel()
        tensors = [z.permute(0, 2, 1, 3).contiguous() for z in (q, k, v, g, b, w)]
        out, final = kernel(*tensors, initial_state=state.float(), output_final_state=True,
                            use_qk_l2norm_in_kernel=True, beta_guard=self.beta_guard)
        return out.permute(0, 2, 1, 3), final

    def forward(self, x, *, return_final_state=False):
        if x.ndim != 3 or x.shape[-1] != self.hidden or x.shape[0] < 1 or x.shape[1] < 1:
            raise ValueError("x must have nonempty [batch, time, hidden] shape")
        batch, length, _ = x.shape
        if self.mode == "shuffled" and batch < 2:
            raise ValueError("shuffled control requires at least two batch rows")
        heads, keys, values = self.heads, self.kdim, self.vdim
        qp, kp, vp = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        if self.conv_size:
            qp, kp, vp = self.q_conv(qp), self.k_conv(kp), self.v_conv(vp)
        else:
            # Retain the old no-convolution pilot's Q/K and value activation.
            vp = F.silu(vp)
        q = qp.view(batch, length, heads, keys).permute(0, 2, 1, 3)
        k = kp.view(batch, length, heads, keys).permute(0, 2, 1, 3)
        v = vp.view(batch, length, heads, values).permute(0, 2, 1, 3)
        b = torch.sigmoid(self.erase_proj(x).view(batch, length, heads, keys)).permute(0, 2, 1, 3)
        w = torch.sigmoid(self.write_proj(x).view(batch, length, heads, values)).permute(0, 2, 1, 3)
        g = (-.02 * F.softplus(self.decay_proj(x).float().view(
            batch, length, heads, keys))).permute(0, 2, 1, 3)
        normalized_q, normalized_k = normalize_qk(q), normalize_qk(k)
        previous_query = torch.cat((torch.zeros_like(normalized_q[:, :, :1]),
                                    normalized_q[:, :, :-1]), dim=2)
        state = torch.zeros(batch, heads, keys, values, device=x.device, dtype=torch.float32)
        gamma = self.log_gamma.clamp(max=math.log(MAX_GAMMA)).exp()
        outs, gates = [], []
        collect = self.diagnostics and not self.training
        diagnostic_rows = {} if collect and self.backend == "reference" else None
        gate_diagnostics = {} if collect else None
        for start in range(0, length, self.chunk):
            stop = min(length, start + self.chunk)
            qq, kk, vv = q[:, :, start:stop], k[:, :, start:stop], v[:, :, start:stop]
            gg, bb, ww = g[:, :, start:stop], b[:, :, start:stop], w[:, :, start:stop]
            original_gates = (bb, ww)
            rho = torch.ones(batch, heads, stop - start, device=x.device, dtype=torch.float32)
            if self.mode != "baseline" or collect:
                memory = state.detach() if self.gate_memory_detach else state
                probe = normalized_k[:, :, start:stop] if self.mode == "keyread" else previous_query[:, :, start:stop]
                compare = vv.roll(1, dims=0) if self.mode == "shuffled" else vv
                with torch.autocast(device_type=x.device.type, enabled=False):
                    read = torch.einsum("bhkv,bhtk->bhtv", memory.float(), probe.float())
                    score = F.cosine_similarity(read, compare.float(), dim=-1, eps=1e-6)
                    if self.mode != "baseline":
                        rho = torch.exp(-gamma.float()[None, :, None] * F.relu(score))
                if collect:
                    for name, value in (("cosine", score), ("read_norm", read.norm(dim=-1)),
                                        ("value_norm", compare.float().norm(dim=-1))):
                        gate_diagnostics.setdefault(name, []).append(value.detach())
                if self.mode == "write":
                    ww = (ww * rho.unsqueeze(-1)).to(ww.dtype)
                elif self.mode != "baseline":
                    bb = (bb * rho.unsqueeze(-1)).to(bb.dtype)
                    ww = (ww * rho.unsqueeze(-1)).to(ww.dtype)
            gates.append(rho.detach())
            if self.backend == "official":
                y, state = self._official_chunk(qq, kk, vv, gg, bb, ww, state)
            else:
                y, state = self._reference_chunk(qq, kk, vv, gg, bb, ww, state,
                    baseline_gates=original_gates, diagnostic_rows=diagnostic_rows)
            outs.append(y)
        combined_gate = torch.cat(gates, dim=2)
        self.last_rho = combined_gate if not self.training else None
        self.last_diagnostics = None
        if collect:
            self.last_diagnostics = {name: torch.cat(rows, dim=2)
                                     for name, rows in gate_diagnostics.items()}
            self.last_diagnostics["rho"] = combined_gate
            if diagnostic_rows is not None:
                self.last_diagnostics.update({name: torch.stack(rows, dim=2)
                                              for name, rows in diagnostic_rows.items()})
        y = torch.cat(outs, dim=2).permute(0, 2, 1, 3).reshape(batch, length, heads * values)
        stats = dict(skip_mean=float(combined_gate.mean()), gamma=float(gamma.detach().mean()),
                     gamma_min=float(gamma.detach().min()), gamma_max=float(gamma.detach().max()))
        result = (self.output(y), stats)
        return (*result, state) if return_final_state else result


def edit_step(state, k, v, b, w, g=None):
    """One RAW GDN-2 edit preserving the caller's dtype for algebraic checks."""
    decay = torch.exp(g) if g is not None else torch.ones_like(k)
    prior = decay[:, None] * state
    return prior + torch.outer(k, w * v - (b * k) @ prior)
