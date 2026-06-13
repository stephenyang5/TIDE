"""
Patch-level TTCN encoder (t-PatchGNN IMTS_Model front block only).

Consumes batched tensors from collate_patches and returns
[B, V, P, hid_dim] patch embeddings plus patch_mask / stay_patch_mask.

Downstream (intra-series Transformer + GNN) should consume this output.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.data.feature_vocab import NUM_FEATURES
from src.models.time_embedding import LearnableTimeEmbedding
from src.models.ttcn import TTCN


class PatchTTCNEncoder(nn.Module):
    def __init__(self, hid_dim: int = 32, te_dim: int = 10) -> None:
        super().__init__()
        if hid_dim < 2:
            raise ValueError("hid_dim must be >= 2 (TTCN uses hid_dim-1 internal units)")
        self.hid_dim = hid_dim
        self.te_dim = te_dim
        self.learnable_te = LearnableTimeEmbedding(te_dim)
        input_dim = 1 + te_dim
        self.ttcn = TTCN(input_dim=input_dim, hid_dim=hid_dim)

    def forward(
        self,
        values: torch.Tensor,
        times: torch.Tensor,
        point_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        values: (B, V, P, L) normalized measurement.
        times: (B, V, P, L) normalized time in [0, 1].
        point_mask: (B, V, P, L) 1 = observed.

        Returns:
            embeddings: (B, V, P, hid_dim)
            patch_mask: (B, V, P) float 1 if patch has any observation (recomputed).
        """
        b, v, p, l = values.shape
        if v != NUM_FEATURES:
            raise ValueError(f"Expected V={NUM_FEATURES}, got {v}")

        vm = point_mask.unsqueeze(-1)
        tt = times.reshape(b * v * p, l, 1)
        te = self.learnable_te(tt)
        xv = values.reshape(b * v * p, l, 1)
        x_int = torch.cat([xv, te], dim=-1)
        mask_x = vm.reshape(b * v * p, l, 1)

        patch_obs = (mask_x.sum(dim=1) > 0).float()
        h = self.ttcn(x_int, mask_x)
        h = torch.cat([h, patch_obs], dim=-1)
        out = h.view(b, v, p, self.hid_dim)

        patch_mask = (point_mask.sum(dim=-1) > 0).float()
        return out, patch_mask
