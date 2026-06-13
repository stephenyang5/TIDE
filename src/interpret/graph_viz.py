"""Adaptive graph (A_p) extraction and visualisation.

The ``TemporalAdaptiveGNNStack`` computes a per-patch, per-batch
adjacency matrix ``adp (B, P, V, V)`` at each layer.  We expose this via
the model's ``_graph_cache`` attribute and average it over (B, P) to
obtain a ``(V, V)`` feature co-activation graph.  Comparing delirium vs
no-delirium patients reveals which feature interactions are delirium-specific.
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

_BLUE   = "#3a7ebf"
_ORANGE = "#e07b39"
_GREEN  = "#3abf7e"
_GRAY   = "#888888"

_GROUP_COLOR = {"chart": _BLUE, "lab": _GREEN, "drug": _ORANGE}

# Short labels for the 57 features (truncated to 12 chars for axis readability)
_SHORT = [n.replace("drug_", "").replace("gcs_", "gcs-")[:12] for n in FEATURE_NAMES]


def _feature_group(name: str) -> str:
    for g, names in FEATURE_GROUPS.items():
        if name in names:
            return g
    return "other"


# ── extraction ───────────────────────────────────────────────────────────────

def extract_adjacency(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    exclude_idxs: list[int] | None = None,
) -> dict[str, np.ndarray]:
    """Collect and average per-layer adjacency matrices from the GNN stack.

    Requires ``model.backbone.stack._graph_cache`` to be a list (enabled by
    setting it to ``[]``; disable by setting it back to ``None``).

    Returns
    -------
    dict with keys:
      "pos"  : (n_layer, V, V) — mean A_p for delirium-positive patients
      "neg"  : (n_layer, V, V) — mean A_p for delirium-negative patients
      "all"  : (n_layer, V, V) — mean A_p over all patients
    """
    stack = model.backbone.stack
    stack._graph_cache = []

    model.eval()

    V = len(FEATURE_NAMES)
    n_layer = stack.n_layer

    sum_pos = np.zeros((n_layer, V, V), dtype=np.float64)
    sum_neg = np.zeros((n_layer, V, V), dtype=np.float64)
    cnt_pos = np.zeros(n_layer, dtype=np.float64)
    cnt_neg = np.zeros(n_layer, dtype=np.float64)

    _excl = exclude_idxs or []
    total = len(loader)
    with torch.no_grad():
        for i, batch in enumerate(loader, 1):
            print(f"  Graph batch {i}/{total} …", end="\r", flush=True)
            stack._graph_cache.clear()

            batch_dev = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            batch_dev = mask_excluded_features(batch_dev, _excl)
            labels = batch_dev["label"].cpu().numpy()  # (B,)
            model(batch_dev)

            # _graph_cache now contains n_layer tensors, each (B, P, V, V)
            for layer_idx, adp in enumerate(stack._graph_cache):
                # Average over P (patch dim): (B, V, V)
                adp_np = adp.numpy().mean(axis=1)  # (B, V, V)

                pos_idx = np.where(labels == 1)[0]
                neg_idx = np.where(labels == 0)[0]

                if len(pos_idx):
                    sum_pos[layer_idx] += adp_np[pos_idx].sum(axis=0)
                    cnt_pos[layer_idx] += len(pos_idx)
                if len(neg_idx):
                    sum_neg[layer_idx] += adp_np[neg_idx].sum(axis=0)
                    cnt_neg[layer_idx] += len(neg_idx)

    print()
    stack._graph_cache = None  # disable capture

    mean_pos = np.zeros_like(sum_pos)
    mean_neg = np.zeros_like(sum_neg)
    for l in range(n_layer):
        if cnt_pos[l] > 0:
            mean_pos[l] = sum_pos[l] / cnt_pos[l]
        if cnt_neg[l] > 0:
            mean_neg[l] = sum_neg[l] / cnt_neg[l]

    total_cnt = cnt_pos + cnt_neg
    mean_all = np.where(
        total_cnt[:, None, None] > 0,
        (sum_pos + sum_neg) / np.maximum(total_cnt[:, None, None], 1),
        0.0,
    )

    return {"pos": mean_pos, "neg": mean_neg, "all": mean_all}


def extract_patient_adjacency(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    exclude_idxs: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-patient mean adjacency over patches and layers.

    Returns
    -------
    graphs : (N, n_layer, V, V) — one graph per patient (mean over patches)
    labels : (N,) int
    probs  : (N,) float predicted probability
    """
    stack = model.backbone.stack
    stack._graph_cache = []
    model.eval()

    graphs: list[np.ndarray] = []
    labels: list[int] = []
    probs: list[float] = []
    _excl = exclude_idxs or []

    with torch.no_grad():
        for batch in loader:
            stack._graph_cache.clear()
            batch_dev = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            batch_dev = mask_excluded_features(batch_dev, _excl)
            logits = model(batch_dev).squeeze(-1)

            # Mean over patches per layer → (B, n_layer, V, V)
            layer_graphs = []
            for adp in stack._graph_cache:
                layer_graphs.append(adp.numpy().mean(axis=1))  # (B, V, V)
            batch_graphs = np.stack(layer_graphs, axis=1)  # (B, n_layer, V, V)

            graphs.append(batch_graphs)
            labels.extend(batch_dev["label"].cpu().tolist())
            probs.extend(torch.sigmoid(logits).cpu().tolist())

    stack._graph_cache = None
    return np.concatenate(graphs, axis=0), np.array(labels), np.array(probs)


def summarize_graph_heterogeneity(
    graphs: np.ndarray,
    output_dir: Path,
    *,
    layer: int = -1,
) -> pd.DataFrame:
    """Edge-wise variance of A_p across patients; save heatmap + top edges."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    g = graphs[:, layer]  # (N, V, V)
    edge_var = g.var(axis=0)
    edge_mean = g.mean(axis=0)

    np.save(output_dir / "graph_edge_variance.npy", edge_var)
    np.save(output_dir / "graph_edge_mean.npy", edge_mean)

    vmax = float(np.percentile(edge_var, 99)) or 1e-6
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    feat_groups = [_feature_group(n) for n in FEATURE_NAMES]
    group_colors = [_GROUP_COLOR.get(g, _GRAY) for g in feat_groups]

    for ax, data, title in [
        (axes[0], edge_mean, f"Mean A_p (layer {layer})"),
        (axes[1], edge_var, f"Cross-patient variance (layer {layer})"),
    ]:
        cmap = "Blues" if title.startswith("Mean") else "Oranges"
        im = ax.imshow(data, cmap=cmap, vmin=0, vmax=vmax if "variance" not in title else float(np.percentile(data, 99)),
                       aspect="auto", origin="upper", interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ticks = np.arange(len(FEATURE_NAMES))
        ax.set_xticks(ticks)
        ax.set_xticklabels(_SHORT, fontsize=4, rotation=90)
        ax.set_yticks(ticks)
        ax.set_yticklabels(_SHORT, fontsize=4)
        for lbl, col in zip(ax.get_yticklabels(), group_colors):
            lbl.set_color(col)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    p = output_dir / f"graph_heterogeneity_layer{layer}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {p}")

    # Top variable edges (upper triangle, exclude diagonal)
    V = len(FEATURE_NAMES)
    rows = []
    for i in range(V):
        for j in range(i + 1, V):
            rows.append({
                "feature_i": FEATURE_NAMES[i],
                "feature_j": FEATURE_NAMES[j],
                "edge_variance": float(edge_var[i, j]),
                "edge_mean": float(edge_mean[i, j]),
            })
    df = pd.DataFrame(rows).sort_values("edge_variance", ascending=False).reset_index(drop=True)
    df.head(30).to_csv(output_dir / "graph_top_variable_edges.csv", index=False)
    return df


def plot_patient_graph_examples(
    graphs: np.ndarray,
    labels: np.ndarray,
    probs: np.ndarray,
    output_dir: Path,
    *,
    layer: int = -1,
    seed: int = 42,
) -> None:
    """Show three patients with high graph dissimilarity (not just class means)."""
    output_dir = Path(output_dir)
    g = graphs[:, layer]  # (N, V, V)
    idx_pos = int(np.where(labels == 1)[0][np.argmax(probs[labels == 1])]) if (labels == 1).any() else 0
    idx_neg = int(np.where(labels == 0)[0][np.argmin(probs[labels == 0])]) if (labels == 0).any() else 1
    mean_g = g.mean(axis=0)
    dist = np.linalg.norm(g - mean_g, axis=(1, 2))
    idx_out = int(np.argmax(dist))

    examples = [
        (idx_neg, "Low-risk patient"),
        (idx_out, "Graph outlier (high ||A − mean||)"),
        (idx_pos, "High-risk delirium patient"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    vmax = float(np.percentile(g, 99)) or 1e-6
    for ax, (idx, title) in zip(axes, examples):
        im = ax.imshow(g[idx], cmap="Blues", vmin=0, vmax=vmax, aspect="auto", origin="upper")
        ax.set_title(f"{title}\nlabel={labels[idx]}  p={probs[idx]:.2f}", fontsize=9)
        ticks = np.arange(len(FEATURE_NAMES))
        ax.set_xticks(ticks)
        ax.set_xticklabels(_SHORT, fontsize=4, rotation=90)
        ax.set_yticks(ticks)
        ax.set_yticklabels(_SHORT, fontsize=4)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"Per-patient adaptive graphs (layer {layer})", fontsize=11)
    fig.tight_layout()
    p = output_dir / f"graph_patient_examples_layer{layer}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {p}")

    np.savez(
        output_dir / "graph_patient_examples.npz",
        indices=np.array([e[0] for e in examples]),
        graphs=g[[e[0] for e in examples]],
        labels=labels[[e[0] for e in examples]],
        probs=probs[[e[0] for e in examples]],
    )


# ── visualisation ─────────────────────────────────────────────────────────────

def plot_adjacency(
    adp_dict: dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    """Save adjacency heatmaps for each GNN layer.

    Produces one file per layer:
      graph_layer0.png, graph_layer1.png, …

    Each file has three panels: delirium, no-delirium, difference.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    adp_pos = adp_dict["pos"]  # (n_layer, V, V)
    adp_neg = adp_dict["neg"]
    n_layer, V, _ = adp_pos.shape

    feat_groups = [_feature_group(n) for n in FEATURE_NAMES]
    group_colors = [_GROUP_COLOR.get(g, _GRAY) for g in feat_groups]

    for layer in range(n_layer):
        pos = adp_pos[layer]  # (V, V)
        neg = adp_neg[layer]
        diff = pos - neg

        vmax_cls = float(np.percentile(np.maximum(pos, neg), 99)) or 1e-6
        vmax_dif = float(np.percentile(np.abs(diff), 99)) or 1e-6

        fig, axes = plt.subplots(1, 3, figsize=(20, 7), sharey=True)

        def _draw(ax, data, title, cmap, vmin, vmax):
            im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax,
                           aspect="auto", origin="upper", interpolation="nearest")
            ax.set_title(title, fontsize=10, pad=4)
            ticks = np.arange(V)
            ax.set_xticks(ticks)
            ax.set_xticklabels(_SHORT, fontsize=4, rotation=90)
            ax.set_yticks(ticks)
            ax.set_yticklabels(_SHORT, fontsize=4)
            # colour y-tick labels by group
            for lbl, col in zip(ax.get_yticklabels(), group_colors):
                lbl.set_color(col)
            # group boundary lines
            boundaries = sorted({feat_groups.index(g) for g in feat_groups})
            seen: set[str] = set()
            for i, g in enumerate(feat_groups):
                if g not in seen:
                    seen.add(g)
                    ax.axhline(i - 0.5, color="white", linewidth=1.2)
                    ax.axvline(i - 0.5, color="white", linewidth=1.2)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        _draw(axes[0], pos,  f"Layer {layer} — Delirium",      "Blues",  0, vmax_cls)
        _draw(axes[1], neg,  f"Layer {layer} — No delirium",   "Blues",  0, vmax_cls)
        _draw(axes[2], diff, f"Layer {layer} — Difference",    "RdBu_r", -vmax_dif, vmax_dif)

        legend_patches = [
            mpatches.Patch(color=_BLUE,   label="Chart"),
            mpatches.Patch(color=_GREEN,  label="Lab"),
            mpatches.Patch(color=_ORANGE, label="Drug"),
        ]
        axes[0].legend(handles=legend_patches, loc="lower right", fontsize=7)

        fig.suptitle(
            f"Adaptive Feature Interaction Graph — GNN layer {layer}",
            fontsize=11, y=1.01,
        )
        fig.tight_layout()

        p = output_dir / f"graph_layer{layer}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {p}")
