"""
Intra-series Transformer over patches + inter-series adaptive GCN (t-PatchGNN ``IMTS_Model`` core).

Input ``x`` is patch embeddings ``(B, V, P, D)`` (e.g. from ``PatchTTCNEncoder``).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.feature_vocab import NUM_FEATURES
from src.models.gcn import GCN
from src.models.positional_encoding import PositionalEncoding


class TemporalAdaptiveGNNStack(nn.Module):
    def __init__(
        self,
        *,
        num_variables: int = NUM_FEATURES,
        d_model: int = 32,
        n_layer: int = 1,
        nhead: int = 1,
        tf_layer: int = 1,
        node_dim: int = 10,
        hop: int = 1,
        dropout: float = 0.1,
        max_patches: int = 512,
        static_supports: list[torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        self.num_variables = num_variables
        self.d_model = d_model
        self.n_layer = n_layer
        self.node_dim = node_dim
        ff = max(4 * d_model, 128)

        self.add_pe = PositionalEncoding(d_model, max_len=max_patches)
        self.transformer_encoder = nn.ModuleList()
        for _ in range(n_layer):
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=ff,
                dropout=dropout,
                batch_first=True,
                activation="relu",
            )
            self.transformer_encoder.append(
                nn.TransformerEncoder(
                    enc_layer, num_layers=tf_layer, enable_nested_tensor=False
                )
            )

        self.static_supports: list[torch.Tensor] = list(static_supports or [])
        self.supports_len = len(self.static_supports) + 1

        self.nodevec1 = nn.Parameter(torch.randn(num_variables, node_dim) * 0.1)
        self.nodevec2 = nn.Parameter(torch.randn(node_dim, num_variables) * 0.1)

        self.nodevec_linear1 = nn.ModuleList()
        self.nodevec_linear2 = nn.ModuleList()
        self.nodevec_gate1 = nn.ModuleList()
        self.nodevec_gate2 = nn.ModuleList()
        for _ in range(n_layer):
            self.nodevec_linear1.append(nn.Linear(d_model, node_dim))
            self.nodevec_linear2.append(nn.Linear(d_model, node_dim))
            self.nodevec_gate1.append(
                nn.Sequential(nn.Linear(d_model + node_dim, 1), nn.Tanh(), nn.ReLU())
            )
            self.nodevec_gate2.append(
                nn.Sequential(nn.Linear(d_model + node_dim, 1), nn.Tanh(), nn.ReLU())
            )

        self.gconv = nn.ModuleList()
        for _ in range(n_layer):
            self.gconv.append(
                GCN(d_model, d_model, dropout, support_len=self.supports_len, order=hop)
            )

        # Set to a list to capture per-layer adjacency matrices during forward();
        # set back to None to disable capture (default).
        self._graph_cache: list[torch.Tensor] | None = None

    def forward(
        self,
        x: torch.Tensor,
        patch_mask: torch.Tensor,
        stay_patch_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        x: (B, V, P, D)
        patch_mask: (B, V, P) 1 if variable observed in patch
        stay_patch_mask: (B, P) 1 if patch within ICU stay (not batch padding)
        """
        b, v, p, d = x.shape
        if v != self.num_variables:
            raise ValueError(f"Expected V={self.num_variables}, got {v}")
        if d != self.d_model:
            raise ValueError(f"Expected D={self.d_model}, got {d}")

        stay = stay_patch_mask.unsqueeze(1).expand(b, v, p).float()
        valid = stay * patch_mask.float()
        key_padding = valid < 0.5
        key_padding_flat = key_padding.reshape(b * v, p)

        # PyTorch TransformerEncoder produces NaN when ALL positions are masked.
        # Pre-compute a safe mask that unmasks one token for those rows.
        all_masked = key_padding_flat.all(dim=-1)          # (B*V,) bool
        if all_masked.any():
            safe_kp = key_padding_flat.clone()
            safe_kp[all_masked, 0] = False                 # unmask first token as anchor
        else:
            safe_kp = key_padding_flat

        for layer in range(self.n_layer):
            if layer > 0:
                x_last = x.clone()

            xv = x.reshape(b * v, p, d)
            xv = self.add_pe(xv)
            xv = self.transformer_encoder[layer](xv, src_key_padding_mask=safe_kp)
            if all_masked.any():
                xv[all_masked] = 0.0
            x = xv.view(b, v, p, d)

            nv1 = self.nodevec1.view(1, 1, v, self.node_dim).expand(b, p, v, self.node_dim)
            nv2 = self.nodevec2.view(1, 1, self.node_dim, v).expand(b, p, self.node_dim, v)

            g1 = self.nodevec_gate1[layer](torch.cat([x, nv1.permute(0, 2, 1, 3)], dim=-1))
            g2 = self.nodevec_gate2[layer](torch.cat([x, nv2.permute(0, 3, 1, 2)], dim=-1))
            x_p1 = g1 * self.nodevec_linear1[layer](x)
            x_p2 = g2 * self.nodevec_linear2[layer](x)
            nv1 = nv1 + x_p1.permute(0, 2, 1, 3)
            nv2 = nv2 + x_p2.permute(0, 2, 3, 1)

            adp = F.softmax(F.relu(torch.matmul(nv1, nv2)), dim=-1)
            if self._graph_cache is not None:
                self._graph_cache.append(adp.detach().cpu())
            supports = [s.to(x.device) for s in self.static_supports] + [adp]

            x_g = self.gconv[layer](x.permute(0, 3, 1, 2), supports)
            x = x_g.permute(0, 2, 3, 1)

            if layer > 0:
                x = x + x_last

        return x
