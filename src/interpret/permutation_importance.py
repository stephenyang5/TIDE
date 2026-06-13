"""Permutation feature importance with repeat-based confidence intervals.

Unlike single-pass ablation (zero a feature in-place), permutation shuffles each
feature's (values, point_mask) tensors across patients, preserving marginal
distributions while breaking the feature–label association.  Repeating the
shuffle ``n_repeats`` times yields a distribution of AUROC drops per feature.
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
from src.interpret.feature_ablation import baseline_auroc, _feature_group

_BLUE = "#3a7ebf"
_ORANGE = "#e07b39"
_GREEN = "#3abf7e"
_GRAY = "#888888"
_GROUP_COLOR = {"chart": _BLUE, "lab": _GREEN, "drug": _ORANGE}


def _eval_auroc(model: nn.Module, loader, device: torch.device, exclude_idxs: list[int]) -> float:
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch_dev = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            batch_dev = mask_excluded_features(batch_dev, exclude_idxs)
            logits = model(batch_dev).squeeze(-1)
            probs.extend(torch.sigmoid(logits).cpu().tolist())
            labels.extend(batch_dev["label"].cpu().tolist())
    y, p = np.array(labels), np.array(probs)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    return float(roc_auc_score(y, p))


def _permute_feature_batch(
    batch: dict,
    feature_idx: int,
    perm: np.ndarray,
) -> dict:
    """Shuffle one feature channel across the batch dimension."""
    vals = batch["values"].clone()
    pm = batch["point_mask"].clone()
    vals[:, feature_idx] = vals[perm, feature_idx]
    pm[:, feature_idx] = pm[perm, feature_idx]
    return {**batch, "values": vals, "point_mask": pm}


def permute_feature_auroc(
    model: nn.Module,
    loader,
    feature_idx: int,
    device: torch.device,
    exclude_idxs: list[int],
    rng: np.random.Generator,
) -> float:
    """AUROC after one random within-batch permutation of *feature_idx*."""
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch_dev = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            batch_dev = mask_excluded_features(batch_dev, exclude_idxs)
            b = batch_dev["values"].shape[0]
            perm = rng.permutation(b)
            batch_dev = _permute_feature_batch(batch_dev, feature_idx, perm)
            logits = model(batch_dev).squeeze(-1)
            probs.extend(torch.sigmoid(logits).cpu().tolist())
            labels.extend(batch_dev["label"].cpu().tolist())
    y, p = np.array(labels), np.array(probs)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    return float(roc_auc_score(y, p))


def run_permutation_importance(
    model: nn.Module,
    loader,
    device: torch.device,
    *,
    ref_auroc: float | None = None,
    exclude_idxs: list[int] | None = None,
    n_repeats: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """Permutation importance with repeat-based 95% CIs on AUROC drop.

    Columns: feature_name, group, mean_drop, std_drop, ci_lo, ci_hi,
             mean_ablated_auroc
    """
    _excl = exclude_idxs or []
    if ref_auroc is None:
        ref_auroc = baseline_auroc(model, loader, device, exclude_idxs=_excl)

    rows: list[dict] = []
    for i, name in enumerate(FEATURE_NAMES):
        if i in _excl:
            continue
        rng = np.random.default_rng(seed + i)
        drops: list[float] = []
        ablated: list[float] = []
        for _ in range(n_repeats):
            auc = permute_feature_auroc(model, loader, i, device, _excl, rng)
            ablated.append(auc)
            drops.append(ref_auroc - auc)
        arr = np.array(drops)
        rows.append({
            "feature_name": name,
            "group": _feature_group(name),
            "mean_drop": float(arr.mean()),
            "std_drop": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "ci_lo": float(np.quantile(arr, 0.025)),
            "ci_hi": float(np.quantile(arr, 0.975)),
            "mean_ablated_auroc": float(np.mean(ablated)),
        })
        print(f"  Permutation {i+1:2d}/{len(FEATURE_NAMES)}  {name:<30s}", end="\r", flush=True)
    print()

    return (
        pd.DataFrame(rows)
        .sort_values("mean_drop", ascending=False)
        .reset_index(drop=True)
    )


def plot_permutation_importance(df: pd.DataFrame, output_dir: Path) -> None:
    """Horizontal bar chart with 95% CI error bars."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, max(6, 0.22 * len(df))))
    y = np.arange(len(df))
    colors = [_GROUP_COLOR.get(g, _GRAY) for g in df["group"]]
    err_lo = df["mean_drop"] - df["ci_lo"]
    err_hi = df["ci_hi"] - df["mean_drop"]
    ax.barh(y, df["mean_drop"], xerr=[err_lo, err_hi], color=colors,
            edgecolor="white", linewidth=0.5, capsize=2, error_kw={"linewidth": 0.8})
    ax.set_yticks(y)
    ax.set_yticklabels(df["feature_name"], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("AUROC drop (permutation importance, 95% CI)", fontsize=9)
    ax.set_title("Permutation Feature Importance", fontsize=11)
    ax.axvline(0, color="black", linewidth=0.8)
    legend_patches = [
        mpatches.Patch(color=_BLUE, label="Chart"),
        mpatches.Patch(color=_GREEN, label="Lab"),
        mpatches.Patch(color=_ORANGE, label="Drug"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=8)
    fig.tight_layout()
    p = output_dir / "permutation_importance.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {p}")
