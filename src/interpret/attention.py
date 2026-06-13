"""Transformer attention-weight extraction and visualisation.

Uses a context-manager approach that monkey-patches every
``nn.MultiheadAttention.forward`` in the model to force
``need_weights=True, average_attn_weights=False`` so per-head weights
are returned and captured, without changing the model's public API.

Flash-attention (PyTorch ≥ 2.0) does not compute weights; if the
captured weight list is empty after a forward pass the extraction
falls back gracefully and the plotting step is skipped.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from src.data.batch_mask import mask_excluded_features

_BLUE = "#3a7ebf"


# ── attention-weight capture context manager ─────────────────────────────────

class AttentionExtractor:
    """Patches all MHA modules in *model* to capture attention weights.

    Usage::

        extractor = AttentionExtractor(model)
        with extractor:
            model(batch)           # weights captured in extractor.weights
        weights = extractor.weights    # list of (B*V, nhead, P, P) tensors
        extractor.clear()

    Can be reused across multiple forward passes; call ``clear()`` between
    passes to reset the weight list.
    """

    def __init__(self, model: nn.Module) -> None:
        self._model = model
        self._weights: list[torch.Tensor] = []
        self._originals: dict[int, tuple[nn.Module, object]] = {}

    # ── context manager protocol ─────────────────────────────────────────

    def __enter__(self) -> "AttentionExtractor":
        # PyTorch ≥ 2.0 has an MHA C++ fast path that bypasses Python module
        # calls entirely, preventing weight capture.  Disable it for the
        # duration of this context so our patched forward() is actually invoked.
        self._prev_fastpath: bool = True
        try:
            self._prev_fastpath = torch.backends.mha.get_fastpath_enabled()
            torch.backends.mha.set_fastpath_enabled(False)
        except AttributeError:
            pass  # older PyTorch without fastpath control

        # Force the math SDP kernel (PyTorch ≥ 2.0) so attention weights are
        # computed even when flash/mem-efficient attention is available.
        self._sdp_ctx = None
        try:
            from torch.nn.attention import sdpa_kernel, SDPBackend  # PyTorch ≥ 2.3
            self._sdp_ctx = sdpa_kernel(SDPBackend.MATH)
            self._sdp_ctx.__enter__()
        except Exception:
            try:
                self._sdp_ctx = torch.backends.cuda.sdp_kernel(
                    enable_flash=False, enable_math=True, enable_mem_efficient=False
                )
                self._sdp_ctx.__enter__()
            except Exception:
                pass  # CPU-only build or older PyTorch

        self._patch_model()
        return self

    def __exit__(self, *args) -> None:
        self._restore_model()
        if self._sdp_ctx is not None:
            try:
                self._sdp_ctx.__exit__(*args)
            except Exception:
                pass
        try:
            torch.backends.mha.set_fastpath_enabled(self._prev_fastpath)
        except AttributeError:
            pass

    # ── public helpers ───────────────────────────────────────────────────

    def clear(self) -> None:
        self._weights.clear()

    @property
    def weights(self) -> list[torch.Tensor]:
        return self._weights

    # ── internal patching ────────────────────────────────────────────────

    def _patch_model(self) -> None:
        storage = self._weights

        for module in self._model.modules():
            if not isinstance(module, nn.MultiheadAttention):
                continue
            if id(module) in self._originals:
                continue  # already patched

            original_forward = module.forward
            self._originals[id(module)] = (module, original_forward)

            def _make_patched(orig, store):
                def patched_forward(query, key, value, *args, **kwargs):
                    kwargs["need_weights"] = True
                    kwargs["average_attn_weights"] = False
                    result = orig(query, key, value, *args, **kwargs)
                    # result: (attn_output, attn_weights)
                    # attn_weights shape: (B*V, nhead, P, P) or None
                    if result[1] is not None:
                        store.append(result[1].detach().cpu())
                    # TransformerEncoderLayer._sa_block takes [0], so returning
                    # the full tuple is safe.
                    return result
                return patched_forward

            module.forward = _make_patched(original_forward, storage)

    def _restore_model(self) -> None:
        for mod_id, (module, orig) in self._originals.items():
            module.forward = orig
        self._originals.clear()


# ── aggregation ──────────────────────────────────────────────────────────────

def aggregate_attention(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    exclude_idxs: list[int] | None = None,
) -> np.ndarray | None:
    """Collect attention weights over the full loader and average.

    Returns
    -------
    attn_mean : np.ndarray of shape (n_calls, nhead, P, P)  or None if no
        weights were captured (e.g. flash attention is active).

    Each forward pass produces ``n_layer * tf_layer`` MHA calls.  We
    accumulate all calls separately so the caller can distinguish layers.
    """
    _excl = exclude_idxs or []
    model.eval()
    extractor = AttentionExtractor(model)

    # Accumulate (n_calls, nhead, P, P) sums and counts
    accumulated: list[np.ndarray] | None = None
    n_batches = 0

    total = len(loader)
    with extractor:
        with torch.no_grad():
            for i, batch in enumerate(loader, 1):
                print(f"  Attention batch {i}/{total} …", end="\r", flush=True)
                extractor.clear()
                batch_dev = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                batch_dev = mask_excluded_features(batch_dev, _excl)
                model(batch_dev)

                if not extractor.weights:
                    continue  # flash attention — no weights returned

                # extractor.weights: list of (B*V, nhead, P, P)
                # Average over B*V for each call, giving (nhead, P, P)
                call_means = [w.mean(dim=0).numpy() for w in extractor.weights]  # list of (nhead, P, P)

                if accumulated is None:
                    accumulated = [np.zeros_like(cm) for cm in call_means]

                for j, cm in enumerate(call_means):
                    if j < len(accumulated):
                        accumulated[j] += cm

                n_batches += 1

    print()

    if accumulated is None or n_batches == 0:
        print("  Warning: no attention weights captured. Flash attention may be active.")
        return None

    attn_mean = np.stack([a / n_batches for a in accumulated], axis=0)  # (n_calls, nhead, P, P)
    return attn_mean


# ── visualisation ─────────────────────────────────────────────────────────────

def plot_attention(
    attn_mean: np.ndarray,
    output_dir: Path,
    patch_hours: int = 8,
) -> None:
    """Plot per-head attention maps as a grid and save to *output_dir*.

    Parameters
    ----------
    attn_mean  : (n_calls, nhead, P, P) array from ``aggregate_attention``
    output_dir : directory to save ``attention_maps.png``
    patch_hours: hours per patch (for axis labels), default 8
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_calls, nhead, P, _ = attn_mean.shape
    patch_labels = [f"h{p*patch_hours}–{(p+1)*patch_hours}" for p in range(P)]

    fig, axes = plt.subplots(
        n_calls, nhead,
        figsize=(3.5 * nhead, 3.0 * n_calls),
        squeeze=False,
    )

    for c in range(n_calls):
        for h in range(nhead):
            ax = axes[c][h]
            im = ax.imshow(
                attn_mean[c, h],
                vmin=0, vmax=attn_mean[c, h].max() or 1.0,
                cmap="Blues",
                aspect="equal",
                origin="upper",
            )
            ax.set_xticks(range(P))
            ax.set_xticklabels(patch_labels, fontsize=7, rotation=30, ha="right")
            ax.set_yticks(range(P))
            ax.set_yticklabels(patch_labels, fontsize=7)
            ax.set_title(f"MHA call {c+1}, head {h+1}", fontsize=8, pad=3)
            ax.set_xlabel("Key patch", fontsize=7)
            if h == 0:
                ax.set_ylabel("Query patch", fontsize=7)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Transformer Attention Weights (averaged over test patients)", fontsize=11, y=1.01)
    fig.tight_layout()

    p = output_dir / "attention_maps.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {p}")
