"""Feature and patch ablation importance for the delirium model.

For each of the 57 features (or 3 time patches) we zero out both the
feature values *and* the observation mask so the model perceives the
feature as completely unobserved.  The AUROC drop from the full-model
baseline quantifies each feature's contribution.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from src.data.batch_mask import mask_excluded_features
from src.data.feature_vocab import FEATURE_NAMES, FEATURE_GROUPS

_BLUE   = "#3a7ebf"
_ORANGE = "#e07b39"
_GREEN  = "#3abf7e"
_GRAY   = "#888888"

_GROUP_COLOR = {"chart": _BLUE, "lab": _GREEN, "drug": _ORANGE}


def _feature_group(name: str) -> str:
    for g, names in FEATURE_GROUPS.items():
        if name in names:
            return g
    return "other"


def _eval_auroc(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    mod_fn=None,
    exclude_idxs: list[int] | None = None,
) -> float:
    """Run inference with optional training-time exclusion, then ``mod_fn``."""
    _excl = exclude_idxs or []
    model.eval()
    all_probs: list[float] = []
    all_labels: list[int] = []
    with torch.no_grad():
        for batch in loader:
            batch_dev = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            batch_dev = mask_excluded_features(batch_dev, _excl)
            if mod_fn is not None:
                batch_dev = mod_fn(batch_dev)
            logits = model(batch_dev).squeeze(-1)
            all_probs.extend(torch.sigmoid(logits).cpu().tolist())
            all_labels.extend(batch_dev["label"].cpu().tolist())
    labels_arr = np.array(all_labels)
    probs_arr  = np.array(all_probs)
    if labels_arr.sum() == 0 or labels_arr.sum() == len(labels_arr):
        return float("nan")
    return float(roc_auc_score(labels_arr, probs_arr))


# ── public API ───────────────────────────────────────────────────────────────

def baseline_auroc(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    exclude_idxs: list[int] | None = None,
) -> float:
    """AUROC on the test set with the same feature exclusion as training (if any)."""
    return _eval_auroc(model, loader, device, exclude_idxs=exclude_idxs)


def ablate_feature(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    feature_idx: int,
    device: torch.device,
    exclude_idxs: list[int] | None = None,
) -> float:
    """AUROC when feature *feature_idx* is fully zeroed (values + mask)."""
    def _zero(batch: dict) -> dict:
        vals = batch["values"].clone()
        pm   = batch["point_mask"].clone()
        vals[:, feature_idx, :, :] = 0.0
        pm  [:, feature_idx, :, :] = 0.0
        return {**batch, "values": vals, "point_mask": pm}

    return _eval_auroc(model, loader, device, mod_fn=_zero, exclude_idxs=exclude_idxs)


def ablate_patch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    patch_idx: int,
    device: torch.device,
    exclude_idxs: list[int] | None = None,
) -> float:
    """AUROC when all features in patch *patch_idx* are zeroed."""
    def _zero(batch: dict) -> dict:
        vals = batch["values"].clone()
        pm   = batch["point_mask"].clone()
        vals[:, :, patch_idx, :] = 0.0
        pm  [:, :, patch_idx, :] = 0.0
        return {**batch, "values": vals, "point_mask": pm}

    return _eval_auroc(model, loader, device, mod_fn=_zero, exclude_idxs=exclude_idxs)


def run_feature_ablation(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    ref_auroc: float | None = None,
    exclude_idxs: list[int] | None = None,
) -> pd.DataFrame:
    """Ablate each feature; return DataFrame sorted by AUROC drop.

    Columns: feature_name, group, ablated_auroc, auroc_drop
    """
    if ref_auroc is None:
        print("  Computing baseline AUROC …")
        ref_auroc = baseline_auroc(model, loader, device, exclude_idxs=exclude_idxs)
        print(f"  Baseline AUROC = {ref_auroc:.4f}")

    rows = []
    for i, name in enumerate(FEATURE_NAMES):
        print(f"  Ablating {i+1:2d}/{len(FEATURE_NAMES)}  {name:<30s}", end="\r", flush=True)
        auc = ablate_feature(model, loader, i, device, exclude_idxs=exclude_idxs)
        rows.append({
            "feature_name":  name,
            "group":         _feature_group(name),
            "ablated_auroc": auc,
            "auroc_drop":    ref_auroc - auc,
        })
    print()

    df = pd.DataFrame(rows).sort_values("auroc_drop", ascending=False).reset_index(drop=True)
    return df


def run_patch_ablation(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    patch_hours: int = 8,
    ref_auroc: float | None = None,
    exclude_idxs: list[int] | None = None,
) -> pd.DataFrame:
    """Ablate each time patch; return DataFrame with AUROC drop per patch."""
    if ref_auroc is None:
        ref_auroc = baseline_auroc(model, loader, device, exclude_idxs=exclude_idxs)

    # Infer number of patches from first batch
    first_batch = next(iter(loader))
    n_patches = first_batch["values"].shape[2]

    rows = []
    for p in range(n_patches):
        start_h = p * patch_hours
        end_h   = start_h + patch_hours
        label   = f"h{start_h}–{end_h}"
        auc = ablate_patch(model, loader, p, device, exclude_idxs=exclude_idxs)
        rows.append({
            "patch_idx":     p,
            "hours":         label,
            "ablated_auroc": auc,
            "auroc_drop":    ref_auroc - auc,
        })

    return pd.DataFrame(rows)


# ── visualisation ─────────────────────────────────────────────────────────────

def plot_feature_importance(
    df_feat: pd.DataFrame,
    df_patch: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Save feature and patch importance bar charts to *output_dir*."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Feature importance (horizontal bars) ──────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 13))

    colors = [_GROUP_COLOR.get(_feature_group(n), _GRAY) for n in df_feat["feature_name"]]
    y = np.arange(len(df_feat))

    bars = ax.barh(y, df_feat["auroc_drop"], color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(df_feat["feature_name"], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("AUROC drop when feature ablated", fontsize=9)
    ax.set_title("Feature Importance (AUROC drop)", fontsize=11)
    ax.axvline(0, color="black", linewidth=0.8)

    legend_patches = [
        mpatches.Patch(color=_BLUE,   label="Chart features"),
        mpatches.Patch(color=_GREEN,  label="Lab features"),
        mpatches.Patch(color=_ORANGE, label="Drug features"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=8)
    fig.tight_layout()

    p = output_dir / "feature_ablation.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {p}")

    # ── Patch importance (vertical bars) ──────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    x = np.arange(len(df_patch))
    ax2.bar(x, df_patch["auroc_drop"], color=_BLUE, edgecolor="white", width=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(df_patch["hours"], fontsize=9)
    ax2.set_ylabel("AUROC drop", fontsize=9)
    ax2.set_title("Temporal Window Importance", fontsize=11)
    ax2.axhline(0, color="black", linewidth=0.8)

    fig2.tight_layout()
    p2 = output_dir / "patch_ablation.png"
    fig2.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved {p2}")
