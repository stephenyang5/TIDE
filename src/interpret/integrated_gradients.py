"""Integrated Gradients attribution over input features × time.

Produces attribution maps for **values** and **point_mask** separately.

**Caveats** (see ``docs/interpretability.md``):
- Values IG uses baseline = all-zeros; LOCF-filled hours still carry gradient
  unless ``point_mask`` is also attributed.
- Mask IG treats the binary mask as a continuous input; attributions reflect
  *observation presence* (charting intensity), not clinical score magnitude.
- Neither path is causal; use permutation importance for robust ranking.

Reference: Sundararajan et al. (2017).
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

from src.data.batch_mask import mask_excluded_features
from src.data.feature_vocab import FEATURE_NAMES, FEATURE_GROUPS

# ── colour palette (matches viz.py) ────────────────────────────────────────
_BLUE   = "#3a7ebf"
_ORANGE = "#e07b39"
_GREEN  = "#3abf7e"
_GRAY   = "#888888"

_GROUP_COLORS = {
    "chart": _BLUE,
    "lab":   _GREEN,
    "drug":  _ORANGE,
}


def _feature_group(name: str) -> str:
    for g, names in FEATURE_GROUPS.items():
        if name in names:
            return g
    return "other"


# ── core computation ────────────────────────────────────────────────────────

def _integrated_gradients_field(
    model: nn.Module,
    batch: dict,
    field: str,
    n_steps: int,
    device: torch.device,
    exclude_idxs: list[int] | None,
) -> np.ndarray:
    """IG w.r.t. ``batch[field]`` (``values`` or ``point_mask``)."""
    model.eval()
    batch_dev = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }
    batch_dev = mask_excluded_features(batch_dev, exclude_idxs or [])
    target: torch.Tensor = batch_dev[field]
    baseline = torch.zeros_like(target)
    accumulated = torch.zeros_like(target)

    for k in range(n_steps):
        alpha = k / max(n_steps - 1, 1)
        interp = (baseline + alpha * (target - baseline)).detach().requires_grad_(True)
        batch_k = {**batch_dev, field: interp}
        logit = model(batch_k).sum()
        logit.backward()
        accumulated = accumulated + interp.grad.detach()  # type: ignore[operator]

    avg_grads = accumulated / n_steps
    return (avg_grads * (target - baseline)).cpu().numpy()


def compute_integrated_gradients(
    model: nn.Module,
    batch: dict,
    n_steps: int = 50,
    device: torch.device = torch.device("cpu"),
    exclude_idxs: list[int] | None = None,
) -> np.ndarray:
    """IG w.r.t. ``values`` only (backward-compatible wrapper).

    Returns attribution shape (B, V, P, L).
    """
    return _integrated_gradients_field(
        model, batch, "values", n_steps, device, exclude_idxs
    )


def compute_integrated_gradients_mask(
    model: nn.Module,
    batch: dict,
    n_steps: int = 50,
    device: torch.device = torch.device("cpu"),
    exclude_idxs: list[int] | None = None,
) -> np.ndarray:
    """IG w.r.t. ``point_mask`` — observation-presence attribution."""
    return _integrated_gradients_field(
        model, batch, "point_mask", n_steps, device, exclude_idxs
    )


def aggregate_ig(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    n_steps: int = 50,
    device: torch.device = torch.device("cpu"),
    exclude_idxs: list[int] | None = None,
    *,
    include_mask: bool = True,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run IG over the loader for values (and optionally point_mask).

    Returns ``(attrs_values, labels)`` or ``(attrs_values, attrs_mask, labels)``.
    """
    vals_list: list[np.ndarray] = []
    mask_list: list[np.ndarray] = []
    labels_list: list[int] = []

    total = len(loader)
    for i, batch in enumerate(loader, 1):
        print(f"  IG batch {i}/{total} …", end="\r", flush=True)
        vals_list.append(
            compute_integrated_gradients(
                model, batch, n_steps=n_steps, device=device, exclude_idxs=exclude_idxs
            )
        )
        if include_mask:
            mask_list.append(
                compute_integrated_gradients_mask(
                    model, batch, n_steps=n_steps, device=device, exclude_idxs=exclude_idxs
                )
            )
        labels_list.extend(batch["label"].tolist())

    print()
    attrs_v = np.concatenate(vals_list, axis=0)
    labels = np.array(labels_list, dtype=int)
    if include_mask:
        return attrs_v, np.concatenate(mask_list, axis=0), labels
    return attrs_v, labels


# ── visualisation ────────────────────────────────────────────────────────────

def plot_ig_heatmap(
    attrs: np.ndarray,
    labels: np.ndarray,
    output_dir: Path,
) -> None:
    """Save IG attribution heatmaps to *output_dir*.

    Produces two files:
      ig_heatmap.png          — overall mean |attribution| across all patients
      ig_heatmap_by_class.png — three-panel: pos mean, neg mean, difference
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    V, P, L = attrs.shape[1], attrs.shape[2], attrs.shape[3]
    n_hours = P * L  # typically 24

    # Reshape (N, V, P, L) → (N, V, H) where H = P*L = 24
    flat = attrs.reshape(len(attrs), V, n_hours)  # (N, V, 24)

    pos_mask = labels == 1
    neg_mask = labels == 0

    mean_all  = np.abs(flat).mean(axis=0)                              # (V, 24)
    mean_pos  = np.abs(flat[pos_mask]).mean(axis=0) if pos_mask.any() else np.zeros((V, n_hours))
    mean_neg  = np.abs(flat[neg_mask]).mean(axis=0) if neg_mask.any() else np.zeros((V, n_hours))
    diff      = mean_pos - mean_neg                                    # (V, 24)

    feat_groups = [_feature_group(n) for n in FEATURE_NAMES]
    group_colors = [_GROUP_COLORS.get(g, _GRAY) for g in feat_groups]

    def _draw_heatmap(ax, data, title, cmap="Blues", center=None, vmax=None):
        im = ax.imshow(
            data,
            aspect="auto",
            origin="upper",
            cmap=cmap,
            vmin=-vmax if center is not None else 0,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_title(title, fontsize=10, pad=4)
        ax.set_xlabel("ICU hour", fontsize=8)
        ax.set_ylabel("")
        ax.set_xticks(np.arange(0, n_hours, 4))
        ax.set_xticklabels(np.arange(0, n_hours, 4), fontsize=7)
        ax.set_yticks(np.arange(V))
        ax.set_yticklabels(FEATURE_NAMES, fontsize=5)
        # Patch group separators
        boundaries = {g: [] for g in ["chart", "lab", "drug"]}
        for i, g in enumerate(feat_groups):
            boundaries[g].append(i)
        for g, idxs in boundaries.items():
            if idxs:
                ax.axhline(min(idxs) - 0.5, color="white", linewidth=1.5)
        # Colour strips on y-axis ticks
        for label_obj, col in zip(ax.get_yticklabels(), group_colors):
            label_obj.set_color(col)
        # Patch grid lines every 8 hours (patch boundary)
        for ph in range(L, n_hours, L):
            ax.axvline(ph - 0.5, color="white", linewidth=1.0, linestyle="--")
        return im

    # ── Figure 1: overall mean ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 12))
    vmax = float(np.percentile(mean_all, 99)) or 1e-6
    im = _draw_heatmap(ax, mean_all, "Mean |IG attribution| — all patients", vmax=vmax)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="|attribution|")
    legend_patches = [
        mpatches.Patch(color=_BLUE,   label="Chart features"),
        mpatches.Patch(color=_GREEN,  label="Lab features"),
        mpatches.Patch(color=_ORANGE, label="Drug features"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=7)
    fig.tight_layout()
    p = output_dir / "ig_heatmap.png"
    fig.savefig(p, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {p}")

    # ── Figure 2: by class + difference ───────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(26, 12), sharey=True)
    vmax_cls = float(np.percentile(np.maximum(mean_pos, mean_neg), 99)) or 1e-6
    vmax_dif = float(np.percentile(np.abs(diff), 99)) or 1e-6

    im0 = _draw_heatmap(axes[0], mean_pos, "Delirium patients", vmax=vmax_cls)
    im1 = _draw_heatmap(axes[1], mean_neg, "No-delirium patients", vmax=vmax_cls)
    im2 = _draw_heatmap(
        axes[2], diff, "Difference (delirium − no-delirium)",
        cmap="RdBu_r", center=0, vmax=vmax_dif,
    )

    plt.colorbar(im0, ax=axes[0], fraction=0.03, pad=0.02, label="|attribution|")
    plt.colorbar(im1, ax=axes[1], fraction=0.03, pad=0.02, label="|attribution|")
    plt.colorbar(im2, ax=axes[2], fraction=0.03, pad=0.02, label="difference")

    fig.tight_layout()
    p2 = output_dir / "ig_heatmap_by_class.png"
    fig.savefig(p2, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {p2}")


def plot_ig_value_vs_mask(
    attrs_values: np.ndarray,
    attrs_mask: np.ndarray,
    output_dir: Path,
) -> None:
    """Side-by-side mean |IG| for values vs point_mask (feature × hour)."""
    output_dir = Path(output_dir)
    n_hours = attrs_values.shape[2] * attrs_values.shape[3]
    mean_val = np.abs(attrs_values.reshape(len(attrs_values), -1, n_hours)).mean(axis=(0, 2))
    mean_msk = np.abs(attrs_mask.reshape(len(attrs_mask), -1, n_hours)).mean(axis=(0, 2))
    # aggregate over hours → per-feature totals
    V = attrs_values.shape[1]
    per_feat_val = np.abs(attrs_values).sum(axis=(2, 3)).mean(axis=0)
    per_feat_msk = np.abs(attrs_mask).sum(axis=(2, 3)).mean(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 10))
    y = np.arange(V)
    axes[0].barh(y, per_feat_val, color=_BLUE, edgecolor="white")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(FEATURE_NAMES, fontsize=5)
    axes[0].invert_yaxis()
    axes[0].set_title("Mean |IG| — values (clinical magnitude)", fontsize=10)
    axes[0].set_xlabel("|attribution| sum over time")

    axes[1].barh(y, per_feat_msk, color=_ORANGE, edgecolor="white")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(FEATURE_NAMES, fontsize=5)
    axes[1].invert_yaxis()
    axes[1].set_title("Mean |IG| — point_mask (observation presence)", fontsize=10)
    axes[1].set_xlabel("|attribution| sum over time")

    fig.tight_layout()
    p = output_dir / "ig_value_vs_mask_features.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {p}")

    pd.DataFrame({
        "feature_name": FEATURE_NAMES,
        "ig_values": per_feat_val,
        "ig_mask": per_feat_msk,
    }).sort_values("ig_values", ascending=False).to_csv(
        output_dir / "ig_value_vs_mask.csv", index=False
    )
