"""Patient embedding extraction and 2-D projection visualisation.

Uses ``DeliriumClassifier.forward_explain()`` to obtain the final patient
representation ``(B, D)`` before the linear head, then projects to 2-D
with t-SNE (always available via scikit-learn) and optionally UMAP
(imported lazily; skipped gracefully if not installed).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.manifold import TSNE

from src.data.batch_mask import mask_excluded_features

_BLUE   = "#3a7ebf"
_ORANGE = "#e07b39"
_GRAY   = "#cccccc"


# ── extraction ───────────────────────────────────────────────────────────────

def extract_embeddings(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    exclude_idxs: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract final patient embeddings, labels, and predicted probabilities.

    Returns
    -------
    embeddings : (N, D)  — pre-classifier representation
    labels     : (N,)    — ground-truth binary labels
    probs      : (N,)    — predicted delirium probabilities
    """
    _excl = exclude_idxs or []
    model.eval()
    all_emb:    list[np.ndarray] = []
    all_labels: list[int]        = []
    all_probs:  list[float]      = []

    total = len(loader)
    with torch.no_grad():
        for i, batch in enumerate(loader, 1):
            print(f"  Embedding batch {i}/{total} …", end="\r", flush=True)
            batch_dev = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            batch_dev = mask_excluded_features(batch_dev, _excl)
            out = model.forward_explain(batch_dev)
            all_emb.append(out["embedding"].cpu().numpy())
            all_labels.extend(batch_dev["label"].cpu().tolist())
            all_probs.extend(torch.sigmoid(out["logit"]).squeeze(-1).cpu().tolist())

    print()
    return (
        np.concatenate(all_emb, axis=0),
        np.array(all_labels, dtype=int),
        np.array(all_probs, dtype=float),
    )


# ── 2-D projection helpers ────────────────────────────────────────────────────

def _scatter(ax, xy, labels, probs, title):
    """Shared scatter-plot helper."""
    pos = labels == 1
    neg = labels == 0

    # Size encodes confidence: larger = model is more certain
    sizes = 4 + 40 * np.abs(probs - 0.5) * 2  # 4..44

    ax.scatter(
        xy[neg, 0], xy[neg, 1],
        c=_BLUE, s=sizes[neg], alpha=0.35, linewidths=0, label="No delirium",
    )
    ax.scatter(
        xy[pos, 0], xy[pos, 1],
        c=_ORANGE, s=sizes[pos], alpha=0.65, linewidths=0, label="Delirium",
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Component 1", fontsize=8)
    ax.set_ylabel("Component 2", fontsize=8)
    ax.legend(fontsize=8, markerscale=1.5)


# ── public API ────────────────────────────────────────────────────────────────

def plot_tsne(
    embeddings: np.ndarray,
    labels: np.ndarray,
    probs: np.ndarray,
    output_dir: Path,
    perplexity: float = 40.0,
    seed: int = 42,
) -> None:
    """t-SNE projection coloured by label; size encodes model confidence.

    Also attempts a UMAP projection if ``umap-learn`` is installed.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── t-SNE ────────────────────────────────────────────────────────────
    print("  Running t-SNE …")
    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, len(embeddings) - 1),
        random_state=seed,
        max_iter=1000,
        init="pca",
        learning_rate="auto",
    )
    xy_tsne = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(7, 6))
    _scatter(ax, xy_tsne, labels, probs, "Patient embeddings — t-SNE")
    fig.tight_layout()
    p = output_dir / "embeddings_tsne.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {p}")

    # ── UMAP (optional) ──────────────────────────────────────────────────
    try:
        import umap  # type: ignore
        print("  Running UMAP …")
        reducer = umap.UMAP(n_components=2, random_state=seed, n_neighbors=30, min_dist=0.1)
        xy_umap = reducer.fit_transform(embeddings)

        fig2, ax2 = plt.subplots(figsize=(7, 6))
        _scatter(ax2, xy_umap, labels, probs, "Patient embeddings — UMAP")
        fig2.tight_layout()
        p2 = output_dir / "embeddings_umap.png"
        fig2.savefig(p2, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"  Saved {p2}")
    except ImportError:
        print("  umap-learn not installed; skipping UMAP (pip install umap-learn)")
