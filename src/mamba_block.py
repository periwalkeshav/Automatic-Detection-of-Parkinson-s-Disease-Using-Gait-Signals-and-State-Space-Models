"""
Mamba / selective state-space (S6) building blocks.

Uses the official CUDA implementation (``mamba-ssm``, Gu & Dao 2023,
https://github.com/state-spaces/mamba) when it is installed and a GPU is
available. Otherwise it falls back to a compact pure-PyTorch implementation of
the same S6 recurrence, so the model always runs. The fallback is mathematically
equivalent but slower than the fused kernel. It offers two scans: "parallel"
(default; Hillis-Steele doubling, log2(L) vectorised steps) and "sequential"
(one step per time point, far less memory for backprop; used by the windowed
Mamba encoder, where hundreds of short sequences are trained at once).

Which path is active is reported by ``mamba_backend()`` and printed once per run.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:                                    # official CUDA implementation
    from mamba_ssm import Mamba as _OfficialMamba
    _HAVE_MAMBA_SSM = True
except Exception:                       # not installed / no CUDA toolchain
    _OfficialMamba = None
    _HAVE_MAMBA_SSM = False


def _use_official() -> bool:
    # The official kernels are CUDA-only; on CPU/MPS use the PyTorch fallback.
    return _HAVE_MAMBA_SSM and torch.cuda.is_available()


def mamba_backend() -> str:
    return "mamba-ssm (official CUDA kernels)" if _use_official() else "pure-PyTorch S6 fallback"


class MambaBlockTorch(nn.Module):
    """
    Pure-PyTorch Mamba block, API-compatible with ``mamba_ssm.Mamba``.

    Input and output are (B, L, d_model). Implements the S6 recurrence

        h_t = exp(Δ_t A) h_{t-1} + Δ_t B_t x_t ,     y_t = C_t h_t + D x_t

    where Δ, B and C are produced from the input (the *selective* part), so the
    state transition is input-dependent rather than fixed as in classical SSMs.
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, dt_rank: int | None = None, scan: str = "parallel"):
        super().__init__()
        if scan not in ("parallel", "sequential"):
            raise ValueError(f"scan must be 'parallel' or 'sequential', got {scan!r}")
        self.scan = scan
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = int(expand * d_model)
        self.dt_rank = dt_rank or max(1, math.ceil(d_model / 16))

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv,
                                groups=self.d_inner, padding=d_conv - 1, bias=True)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # S4D-real initialisation: A = -(1 … d_state), kept in log space.
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    @staticmethod
    def _parallel_scan(dA, dBx):
        """
        Solve the linear recurrence h_t = dA_t · h_{t-1} + dBx_t along dim 1.

        Hillis–Steele associative scan: the pairs (a, b) compose as
        (a1, b1) ∘ (a2, b2) = (a1·a2, a2·b1 + b2), so the whole scan is done in
        ⌈log2 L⌉ vectorised doubling steps instead of L sequential ones. That is
        what makes the pure-PyTorch path usable without the fused CUDA kernel.
        Numerically safe: the decay factors lie in (0, 1), so products only shrink.
        """
        A, B = dA, dBx
        step = 1
        L = A.shape[1]
        while step < L:
            A_prev = F.pad(A[:, :-step], (0, 0, 0, 0, step, 0), value=1.0)
            B_prev = F.pad(B[:, :-step], (0, 0, 0, 0, step, 0), value=0.0)
            B = B + A * B_prev
            A = A * A_prev
            step *= 2
        return B                                               # h_t for all t

    @staticmethod
    def _sequential_scan(dA, dBx, C):
        """
        The same recurrence as ``_parallel_scan``, one time step at a time, with
        the read-out y_t = C_t h_t folded into the loop.

        The parallel scan keeps several full (b, l, d_inner, d_state) tensors per
        doubling step for the backward pass; for a MIL batch of ~200 short
        windows that is ~20 GB. Here only one (b, d_inner, d_state) state per
        step is kept, and for short sequences the l-step loop is cheap.
        """
        h = torch.zeros_like(dBx[:, 0])                        # (b, d_inner, d_state)
        ys = []
        for t in range(dA.shape[1]):
            h = dA[:, t] * h + dBx[:, t]
            ys.append(torch.einsum("bdn,bn->bd", h, C[:, t]))
        return torch.stack(ys, dim=1)                          # (b, l, d_inner)

    def _selective_scan(self, x, dt, B, C):
        """x, dt: (b, l, d_inner);  B, C: (b, l, d_state)  ->  (b, l, d_inner)."""
        A = -torch.exp(self.A_log)                             # (d_inner, d_state)
        dA = torch.exp(dt.unsqueeze(-1) * A)                   # (b, l, d_inner, d_state)
        dBx = dt.unsqueeze(-1) * B.unsqueeze(2) * x.unsqueeze(-1)

        if self.scan == "sequential":
            y = self._sequential_scan(dA, dBx, C)
        else:
            h = self._parallel_scan(dA, dBx)                   # (b, l, d_inner, d_state)
            y = torch.einsum("bldn,bln->bld", h, C)
        return y + x * self.D

    def forward(self, x):                                      # (b, l, d_model)
        l = x.shape[1]
        x_in, z = self.in_proj(x).chunk(2, dim=-1)

        x_in = self.conv1d(x_in.transpose(1, 2))[:, :, :l].transpose(1, 2)
        x_in = F.silu(x_in)

        dbc = self.x_proj(x_in)
        dt, B, C = torch.split(dbc, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))

        y = self._selective_scan(x_in, dt, B, C)
        return self.out_proj(y * F.silu(z))                    # gated output


def make_mamba(d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2,
               scan: str = "parallel"):
    """One Mamba mixer: official CUDA block when usable, else the PyTorch one.
    `scan` selects the fallback's scan and is ignored by the official kernel."""
    if _use_official():
        return _OfficialMamba(d_model=d_model, d_state=d_state,
                              d_conv=d_conv, expand=expand)
    return MambaBlockTorch(d_model, d_state=d_state, d_conv=d_conv, expand=expand,
                           scan=scan)


class BiMambaLayer(nn.Module):
    """
    Pre-norm residual Mamba layer, optionally bidirectional.

    Gait is not causal — a full stride is visible inside the window — so by
    default the sequence is scanned in both directions and the two states are
    summed, as in bidirectional SSM variants for non-causal signals.
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, dropout: float = 0.0, bidirectional: bool = True,
                 scan: str = "parallel"):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fwd = make_mamba(d_model, d_state, d_conv, expand, scan=scan)
        self.bwd = (make_mamba(d_model, d_state, d_conv, expand, scan=scan)
                    if bidirectional else None)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                                      # (b, l, d_model)
        h = self.norm(x)
        y = self.fwd(h)
        if self.bwd is not None:
            y = y + self.bwd(h.flip(1)).flip(1)
        return x + self.drop(y)                                # residual
