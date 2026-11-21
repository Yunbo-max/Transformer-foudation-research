"""SG-Delta with one content memory and causal historical-query statistics.

Key-major content state: S[..., K, V]. The auxiliary covariance C[..., K, K]
is also persistent within a sequence and must be included in state accounting.
This readable reference has no fused kernel or streaming-cache implementation.
The joint recurrence and its defaults are unchanged; see MATH.md for the exact
objective and the scope of its stability statement.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


ARMS = (
    "isotropic", "query-write", "query-decay", "query-joint",
    "key-joint", "channel-decay", "reverse-joint",
)
GEOMETRY_SCALES = ("raw", "dimension")


def _work_dtype(*tensors):
    """Accumulate in FP32, preserving explicit FP64 mathematical checks."""
    if not all(t.is_floating_point() for t in tensors):
        raise TypeError("SG-Delta requires real floating-point tensors")
    return torch.float64 if any(t.dtype == torch.float64 for t in tensors) else torch.float32


def _positive(name, value, allow_zero=False):
    if not math.isfinite(value) or (value < 0 if allow_zero else value <= 0):
        relation = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {relation}")


def query_geometry(covariance, *, lam=1.0, mu=8.0):
    """Return (H, P, chol(H)) from a symmetric PSD historical covariance.

    H = lam*I + mu*C and P = I - lam*H^{-1}. Symmetrization only removes
    floating-point asymmetry; there is no eigenvalue clipping, jitter retry,
    or silent change to lam. Invalid/non-PD precision fails at Cholesky.
    Leading dimensions are independent sequences/heads.
    """
    _positive("lam", lam)
    _positive("mu", mu, allow_zero=True)
    if covariance.ndim < 2 or covariance.shape[-2] != covariance.shape[-1]:
        raise ValueError("covariance must have shape [..., K, K]")
    if covariance.shape[-1] < 1:
        raise ValueError("key dimension must be positive")
    dtype = _work_dtype(covariance)
    with torch.autocast(device_type=covariance.device.type, enabled=False):
        C = covariance.to(dtype)
        C = 0.5 * (C + C.transpose(-1, -2))
        eye = torch.eye(C.shape[-1], dtype=dtype, device=C.device).expand_as(C)
        precision = lam * eye + mu * C
        factor = torch.linalg.cholesky(precision)
        inverse = torch.cholesky_solve(eye, factor)
        protection = eye - lam * inverse
        protection = 0.5 * (protection + protection.transpose(-1, -2))
    return precision, protection, factor


def protected_write_directions(keys, factor, *, tau=1.0):
    """Solve for a whole chunk: keys[..., L, K], factor[..., K, K]."""
    _positive("tau", tau)
    if keys.ndim < 2 or factor.ndim < 2:
        raise ValueError("keys/factor must include token and matrix dimensions")
    if (keys.shape[:-2] != factor.shape[:-2]
            or factor.shape[-2] != factor.shape[-1]
            or keys.shape[-1] != factor.shape[-1]):
        raise ValueError("keys [..., L, K] and factor [..., K, K] must align")
    dtype = _work_dtype(keys, factor)
    with torch.autocast(device_type=keys.device.type, enabled=False):
        k = keys.to(dtype)
        u = torch.cholesky_solve(k.transpose(-1, -2), factor.to(dtype)).transpose(-1, -2)
        direction = u / (tau + (k * u).sum(dim=-1, keepdim=True))
    return direction


def update_query_covariance(covariance, queries, *, ema=0.98, detach_statistics=True):
    """Consume a completed chunk, never the chunk currently being processed.

    C_new = ema**L*C + (1-ema)*sum_j ema**(L-1-j)*q_j*q_j^T.
    queries has shape [..., L, K]. Every projected token query contributes;
    this is usage geometry, not a label or a successful-retrieval detector.
    """
    if not math.isfinite(ema) or not 0 <= ema < 1:
        raise ValueError("ema must satisfy 0 <= ema < 1")
    if (covariance.ndim < 2 or queries.ndim < 2
            or covariance.shape[:-2] != queries.shape[:-2]
            or covariance.shape[-2] != covariance.shape[-1]
            or covariance.shape[-1] != queries.shape[-1]
            or queries.shape[-2] < 1):
        raise ValueError("covariance [..., K, K] and nonempty queries [..., L, K] must align")
    dtype = _work_dtype(covariance, queries)
    with torch.autocast(device_type=queries.device.type, enabled=False):
        C, q = covariance.to(dtype), queries.to(dtype)
        if detach_statistics:
            C, q = C.detach(), q.detach()
        length = q.shape[-2]
        ages = torch.arange(length - 1, -1, -1, device=q.device)
        weights = (1 - ema) * torch.as_tensor(ema, dtype=dtype, device=q.device).pow(ages)
        updated = ema**length * C + q.transpose(-1, -2) @ (q * weights[:, None])
        updated = 0.5 * (updated + updated.transpose(-1, -2))
    return updated


def sg_delta_step(state, key, value, decay, direction):
    """Pure state update using a precomputed chunk geometry/write direction.

    state [..., K, V], key/direction [..., K], value [..., V], decay [..., K, K].
    Reading the residual AFTER decay is required by the objective in MATH.md.
    """
    if (state.ndim < 2 or key.shape != state.shape[:-1]
            or direction.shape != key.shape
            or value.shape != state.shape[:-2] + (state.shape[-1],)
            or decay.shape != state.shape[:-2] + (state.shape[-2], state.shape[-2])):
        raise ValueError("state, key, value, decay and direction dimensions do not align")
    dtype = _work_dtype(state, key, value, decay, direction)
    with torch.autocast(device_type=state.device.type, enabled=False):
        before = decay.to(dtype) @ state.to(dtype)
        residual = value.to(dtype) - (before * key.to(dtype).unsqueeze(-1)).sum(dim=-2)
        updated = before + direction.to(dtype).unsqueeze(-1) * residual.unsqueeze(-2)
    return updated


class CausalDepthwiseProjection(nn.Module):
    """Optional local q/k/v feature mixing; no future or symmetric padding."""

    def __init__(self, channels, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(channels, channels, kernel_size, groups=channels, bias=False)

    def forward(self, projected):
        channels_first = projected.transpose(1, 2)
        left_padded = F.pad(channels_first, (self.kernel_size - 1, 0))
        return F.silu(self.conv(left_padded).transpose(1, 2))


class SGDeltaMixer(nn.Module):
    def __init__(self, hidden, heads=2, key_dim=16, value_dim=None, chunk=16,
                 mode="query-joint", mu=8.0, lam=1.0, tau=1.0, ema=0.98,
                 detach_statistics=True, geometry_scale="raw", conv_size=0):
        super().__init__()
        value_dim = key_dim if value_dim is None else value_dim
        if any(not isinstance(n, int) or n < 1 for n in (hidden, heads, key_dim, value_dim, chunk)):
            raise ValueError("hidden, heads, key_dim, value_dim and chunk must be positive integers")
        if mode not in ARMS or geometry_scale not in GEOMETRY_SCALES:
            raise ValueError("unknown method or geometry scaling mode")
        if not isinstance(conv_size, int) or conv_size < 0:
            raise ValueError("conv_size must be a nonnegative integer; 0 disables convolution")
        _positive("mu", mu, allow_zero=True)
        _positive("lam", lam)
        _positive("tau", tau)
        if not math.isfinite(ema) or not 0 <= ema < 1:
            raise ValueError("ema must satisfy 0 <= ema < 1")
        self.heads, self.kdim, self.vdim = heads, key_dim, value_dim
        self.chunk, self.mode = chunk, mode
        self.mu, self.lam, self.tau, self.ema = mu, lam, tau, ema
        self.detach_statistics = detach_statistics
        self.geometry_scale = geometry_scale
        self.geometry_scale_factor = key_dim if geometry_scale == "dimension" else 1
        self.effective_mu = mu * self.geometry_scale_factor
        _positive("effective_mu", self.effective_mu, allow_zero=True)
        self.conv_size = conv_size
        self.q_proj = nn.Linear(hidden, heads * key_dim, bias=False)
        self.k_proj = nn.Linear(hidden, heads * key_dim, bias=False)
        self.v_proj = nn.Linear(hidden, heads * value_dim, bias=False)
        self.output = nn.Linear(heads * value_dim, hidden, bias=False)
        self.fast_gate = nn.Linear(hidden, heads)
        self.slow_gate = nn.Linear(hidden, heads)
        self.channel_gate = nn.Linear(hidden, heads * key_dim)
        nn.init.constant_(self.fast_gate.bias, 1.0)
        nn.init.constant_(self.slow_gate.bias, 2.0)
        # Keep the original initialization/parameter stream when conv_size=0.
        self.q_conv = CausalDepthwiseProjection(heads * key_dim, conv_size) if conv_size else nn.Identity()
        self.k_conv = CausalDepthwiseProjection(heads * key_dim, conv_size) if conv_size else nn.Identity()
        self.v_conv = CausalDepthwiseProjection(heads * value_dim, conv_size) if conv_size else nn.Identity()

    def forward(self, x):
        if x.ndim != 3 or x.shape[1] < 1:
            raise ValueError("x must have shape [B, T, hidden] with T >= 1")
        B, T, _ = x.shape
        H, K, V = self.heads, self.kdim, self.vdim
        q_projected = self.q_conv(self.q_proj(x))
        k_projected = self.k_conv(self.k_proj(x))
        v_projected = self.v_conv(self.v_proj(x))
        fast_logits = self.fast_gate(x)
        slow_logits = self.slow_gate(x)
        channel_logits = self.channel_gate(x)
        dtype = _work_dtype(x, q_projected, k_projected, v_projected)
        use_geometry = self.mode in (
            "query-write", "query-decay", "query-joint", "key-joint", "reverse-joint",
        )
        protected_write = self.mode in (
            "query-write", "query-joint", "key-joint", "reverse-joint",
        )

        # AMP may be used for projections, but not for normalization, covariance,
        # Cholesky, residuals, state accumulation or query readout.
        with torch.autocast(device_type=x.device.type, enabled=False):
            q = F.normalize(q_projected.to(dtype).view(B, T, H, K), dim=-1, eps=1e-6).permute(0, 2, 1, 3)
            k = F.normalize(k_projected.to(dtype).view(B, T, H, K), dim=-1, eps=1e-6).permute(0, 2, 1, 3)
            v = v_projected.to(dtype).view(B, T, H, V).permute(0, 2, 1, 3)
            fast = (0.86 + 0.11 * torch.sigmoid(fast_logits.to(dtype))).permute(0, 2, 1)
            slow = fast + (1 - fast) * torch.sigmoid(slow_logits.to(dtype)).permute(0, 2, 1)
            channel = (0.86 + 0.13 * torch.sigmoid(channel_logits.to(dtype).view(B, T, H, K))).permute(0, 2, 1, 3)
            state = torch.zeros(B, H, K, V, dtype=dtype, device=x.device)
            C = torch.zeros(B, H, K, K, dtype=dtype, device=x.device)
            eye = torch.eye(K, dtype=dtype, device=x.device).view(1, 1, K, K)
            outputs, cosines, geometry_means, decay_means = [], [], [], []

            for start in range(0, T, self.chunk):
                stop = min(T, start + self.chunk)
                Cprior = C.detach() if self.detach_statistics else C
                _, P, factor = query_geometry(
                    Cprior, lam=self.lam, mu=self.effective_mu if use_geometry else 0.0,
                )
                geometry_means.append(P.diagonal(dim1=-2, dim2=-1).mean())
                chunk_keys = k[:, :, start:stop]
                if protected_write:
                    direction = protected_write_directions(chunk_keys, factor, tau=self.tau)
                else:
                    u = chunk_keys / self.lam
                    direction = u / (self.tau + (chunk_keys * u).sum(dim=-1, keepdim=True))
                cosines.append(F.cosine_similarity(direction, chunk_keys, dim=-1).mean())

                for j, t in enumerate(range(start, stop)):
                    ft, st = fast[:, :, t], slow[:, :, t]
                    if self.mode in ("query-decay", "query-joint", "key-joint", "reverse-joint"):
                        protect = eye - P if self.mode == "reverse-joint" else P
                        D = ft[..., None, None] * eye + (st - ft)[..., None, None] * protect
                    elif self.mode == "channel-decay":
                        D = torch.diag_embed(channel[:, :, t])
                    else:
                        D = ft[..., None, None] * eye
                    state = sg_delta_step(state, k[:, :, t], v[:, :, t], D, direction[:, :, j])
                    outputs.append((state * q[:, :, t].unsqueeze(-1)).sum(dim=-2))
                    decay_means.append(D.diagonal(dim1=-2, dim2=-1).mean())

                if use_geometry:
                    observations = k[:, :, start:stop] if self.mode == "key-joint" else q[:, :, start:stop]
                    C = update_query_covariance(
                        Cprior, observations, ema=self.ema, detach_statistics=self.detach_statistics,
                    )

            out = torch.stack(outputs, dim=2).permute(0, 2, 1, 3).reshape(B, T, H * V)
            fraction = float(torch.stack(geometry_means).mean().detach())
            stats = dict(
                write_cosine=float(torch.stack(cosines).mean().detach()),
                query_geometry_fraction=fraction,
                importance_fraction=fraction,  # Compatibility alias; NOT semantic importance.
                fast_decay=float(fast.mean().detach()),
                slow_decay=float(slow.mean().detach()),
                actual_mean_decay=float(torch.stack(decay_means).mean().detach()),
                geometry_scale_factor=self.geometry_scale_factor,
                effective_mu=self.effective_mu,
                content_state_elements_per_sequence=H * K * V,
                covariance_state_elements_per_sequence=H * K * K if use_geometry else 0,
                # The reference also allocates a zero covariance buffer for
                # isotropic arms; it carries no observations in those arms.
                reference_covariance_buffer_elements_per_sequence=H * K * K,
                conv_size=self.conv_size,
            )
        return self.output(out.to(v_projected.dtype)), stats
