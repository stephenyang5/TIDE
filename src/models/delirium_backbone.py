"""
Full backbone: TTCN patch encoder + temporal/adaptive GCN stack.
Also contains DeliriumClassifier: backbone + masked mean pooling + binary head.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from src.models.patch_encoder import PatchTTCNEncoder
from src.models.temporal_adaptive_stack import TemporalAdaptiveGNNStack


class DeliriumTPatchBackbone(nn.Module):
    def __init__(
        self,
        *,
        hid_dim: int = 32,
        te_dim: int = 10,
        n_layer: int = 1,
        nhead: int = 1,
        tf_layer: int = 1,
        node_dim: int = 10,
        hop: int = 1,
        dropout: float = 0.1,
        max_patches: int = 512,
    ) -> None:
        super().__init__()
        self.patch_encoder = PatchTTCNEncoder(hid_dim=hid_dim, te_dim=te_dim)
        self.stack = TemporalAdaptiveGNNStack(
            d_model=hid_dim,
            n_layer=n_layer,
            nhead=nhead,
            tf_layer=tf_layer,
            node_dim=node_dim,
            hop=hop,
            dropout=dropout,
            max_patches=max_patches,
        )

    def forward(self, batch: dict[str, Any]):
        """
        batch from ``collate_patches``: ``values``, ``times``, ``point_mask``, ``stay_patch_mask``.
        Returns patch-level hidden states ``(B, V, P, D)`` for a future classification head.
        """
        z, pm = self.patch_encoder(
            batch["values"], batch["times"], batch["point_mask"]
        )
        return self.stack(z, pm, batch["stay_patch_mask"])


class DeliriumClassifier(nn.Module):
    """T-PatchGNN backbone + masked mean pooling + binary classification head.

    Pooling strategy (paper-aligned):
      1. ``DeliriumTPatchBackbone`` → ``(B, V, P, D)``
      2. Masked mean over patch dim using ``stay_patch_mask`` → ``(B, V, D)``
      3. Mean over variable dim → ``(B, D)``
      4. Dropout + ``Linear(D, 1)`` → logit ``(B, 1)``

    Use with ``BCEWithLogitsLoss``.  The batch dict must already be on the
    target device when passed to ``forward``.
    """

    def __init__(
        self,
        *,
        hid_dim: int = 32,
        te_dim: int = 10,
        n_layer: int = 2,
        nhead: int = 4,
        tf_layer: int = 2,
        node_dim: int = 10,
        hop: int = 1,
        dropout: float = 0.1,
        max_patches: int = 512,
    ) -> None:
        super().__init__()
        self.backbone = DeliriumTPatchBackbone(
            hid_dim=hid_dim,
            te_dim=te_dim,
            n_layer=n_layer,
            nhead=nhead,
            tf_layer=tf_layer,
            node_dim=node_dim,
            hop=hop,
            dropout=dropout,
            max_patches=max_patches,
        )
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(hid_dim, 1)

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        h = self.backbone(batch)  # (B, V, P, D)

        # Masked mean over patch dim — stay_patch_mask (B, P) marks valid patches
        spm = batch["stay_patch_mask"].unsqueeze(1).unsqueeze(-1)  # (B, 1, P, 1)
        valid_count = spm.sum(dim=2).clamp(min=1.0)               # (B, 1, 1)
        h_pooled = (h * spm).sum(dim=2) / valid_count             # (B, V, D)

        # Mean over variable dim
        h_pooled = h_pooled.mean(dim=1)  # (B, D)

        return self.classifier(self.drop(h_pooled))  # (B, 1)

    def forward_explain(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Like forward() but also returns intermediate representations for interpretability.

        Returns a dict with:
          logit            (B, 1)   — raw logit (apply sigmoid for probability)
          patch_embeddings (B,V,P,D) — contextual patch-level hidden states
          var_embeddings   (B,V,D)  — after masked pooling over patches
          embedding        (B,D)    — final patient representation before classifier
        """
        h = self.backbone(batch)  # (B, V, P, D)

        spm = batch["stay_patch_mask"].unsqueeze(1).unsqueeze(-1)  # (B, 1, P, 1)
        valid_count = spm.sum(dim=2).clamp(min=1.0)               # (B, 1, 1)
        h_var = (h * spm).sum(dim=2) / valid_count                 # (B, V, D)
        h_pat = h_var.mean(dim=1)                                  # (B, D)
        logit = self.classifier(self.drop(h_pat))                  # (B, 1)

        return {
            "logit":            logit,
            "patch_embeddings": h,
            "var_embeddings":   h_var,
            "embedding":        h_pat,
        }
