"""Transformable Time-aware Convolution (TTCN) from t-PatchGNN.

Ported from tPatchGNN/model/tPatchGNN.py (TTCN and filter generator MLP).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TTCN(nn.Module):
    """Aggregates irregular observations inside a patch via learned time-varying filters."""

    def __init__(self, input_dim: int, hid_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.ttcn_dim = hid_dim - 1
        d_in = input_dim
        d_h = self.ttcn_dim
        self.filter_generators = nn.Sequential(
            nn.Linear(d_in, d_h, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(d_h, d_h, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(d_h, d_in * d_h, bias=True),
        )
        self.t_bias = nn.Parameter(torch.randn(1, d_h))

    def forward(self, x_int: torch.Tensor, mask_x: torch.Tensor) -> torch.Tensor:
        """
        x_int: (N, L, F_in) value + time-embed concatenated per timestep.
        mask_x: (N, L, 1) observation mask.
        Returns: (N, ttcn_dim) after ReLU (+ bias); caller appends patch-level mask.
        """
        n, lx, _ = mask_x.shape
        filt = self.filter_generators(x_int)
        filt_mask = filt * mask_x + (1.0 - mask_x) * (-1e8)
        filt_seqnorm = F.softmax(filt_mask.clamp(min=-80.0), dim=-2)
        filt_seqnorm = filt_seqnorm.view(n, lx, self.ttcn_dim, -1)
        x_broad = x_int.unsqueeze(-2).expand(-1, -1, self.ttcn_dim, -1)
        ttcn_out = torch.sum(torch.sum(x_broad * filt_seqnorm, dim=-3), dim=-1)
        return torch.relu(ttcn_out + self.t_bias)
