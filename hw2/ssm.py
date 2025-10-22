# Insert this into model.py (or a new module and import).
# Minimal lightweight SSM-like layer (depthwise causal conv per channel)
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class SmallSSM(nn.Module):
    """
    Lightweight SSM-ish layer implemented as a learnable causal depthwise 1D conv.
    Input: (B, T, C)
    Output: (B, T, C)
    This is intentionally simple: a per-channel causal FIR kernel of length L.
    """
    def __init__(self, dim, L=64, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.L = L
        # kernel per channel: shape (dim, L)
        # We'll store as parameter and reshape to conv weight shape (dim, 1, L)
        self.kernel = nn.Parameter(torch.randn(dim, L) * 0.02)
        # optional gating / residual projection for expressivity
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, C)
        B, T, C = x.size()
        assert C == self.dim, "channel mismatch"
        # prepare for depthwise conv: (B, C, T)
        x_t = x.transpose(1, 2)  # (B, C, T)
        # pad on the left to make conv causal
        pad = (self.L - 1, 0)
        x_padded = F.pad(x_t, pad)  # (B, C, T + L - 1)
        # conv weight shape: (out_channels, in_channels/groups, kernel_size)
        # For depthwise conv, out_channels = C, in_channels = C, groups = C, weight shape (C, 1, L)
        weight = self.kernel.view(C, 1, self.L)
        out = F.conv1d(x_padded, weight=weight, bias=None, groups=C)  # (B, C, T)
        out = out.transpose(1, 2)  # (B, T, C)
        out = self.proj(out)      # small linear mixing (like attention output projection)
        out = self.dropout(out)
        return out


class SSMAttention(nn.Module):
    """
    Replacement for causal self-attention using a small SSM-ish layer.
    This keeps the same Block interface: input -> LayerNorm -> SSMAttention -> residual
    """
    def __init__(self, config):
        # config assumed same interface as nanoGPT: n_embd, block_size, dropout
        super().__init__()
        n_embd = config.n_embd
        block_size = getattr(config, "block_size", 1024)
        L = min(128, block_size)   # kernel length; tune down for memory/speed
        self.ln = nn.LayerNorm(n_embd)
        self.ssm = SmallSSM(n_embd, L=L, dropout=config.dropout)
        # optionally a small linear gating like attention's out projection (already in SmallSSM)
        # nothing else required

    def forward(self, x):
        # x: (B, T, C)
        x_ln = self.ln(x)
        ssm_out = self.ssm(x_ln)   # (B, T, C)
        return ssm_out