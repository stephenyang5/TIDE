"""Shared evaluation metrics: bootstrap CIs and calibration.

Centralizes metric helpers so the deep model, baselines, and notebooks report
the same numbers the same way.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def bootstrap_ci(
    labels: np.ndarray,
    probs: np.ndarray,
    *,
    n_iter: int = 200,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, tuple[float, float]]:
    """95% CIs for AUROC and AUPRC via bootstrap resampling.

    Degenerate (single-class) resamples are skipped. Returns
    auroc: (lo, hi), auprc: (lo, hi), (nan, nan) if too few valid
    resamples accumulate.
    """
    rng = np.random.default_rng(seed)
    n = len(labels)
    aurocs: list[float] = []
    auprcs: list[float] = []
    for _ in range(n_iter):
        idx = rng.integers(0, n, size=n)
        y_b, p_b = labels[idx], probs[idx]
        if y_b.sum() == 0 or y_b.sum() == len(y_b):
            continue
        aurocs.append(float(roc_auc_score(y_b, p_b)))
        auprcs.append(float(average_precision_score(y_b, p_b)))
    if len(aurocs) < 10:
        nan2 = (float("nan"), float("nan"))
        return {"auroc": nan2, "auprc": nan2}
    lo_q, hi_q = alpha / 2, 1 - alpha / 2
    return {
        "auroc": (float(np.quantile(aurocs, lo_q)), float(np.quantile(aurocs, hi_q))),
        "auprc": (float(np.quantile(auprcs, lo_q)), float(np.quantile(auprcs, hi_q))),
    }


def brier(labels: np.ndarray, probs: np.ndarray) -> float:
    """Brier score (lower is better - 0 = perfect calibration + accuracy)."""
    return float(brier_score_loss(labels, probs))


def reliability_curve(
    labels: np.ndarray,
    probs: np.ndarray,
    *,
    n_bins: int = 10,
) -> dict[str, np.ndarray]:
    """Equal-width reliability curve data for a calibration plot.

    Returns arrays mean_pred, frac_pos
    (observed positive fraction per bin), and count (n per bin). 
    Empty bins are dropped.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(probs, bins) - 1, 0, n_bins - 1)
    mean_pred, frac_pos, count = [], [], []
    for b in range(n_bins):
        m = idx == b
        c = int(m.sum())
        if c == 0:
            continue
        mean_pred.append(float(probs[m].mean()))
        frac_pos.append(float(labels[m].mean()))
        count.append(c)
    return {
        "mean_pred": np.asarray(mean_pred),
        "frac_pos": np.asarray(frac_pos),
        "count": np.asarray(count),
    }


def summarize(labels: np.ndarray, probs: np.ndarray, *, n_boot: int = 200, seed: int = 42) -> dict:
    """One-stop metric dict: AUROC, AUPRC, Brier, prevalence, bootstrap CIs."""
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs, dtype=float)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return {"auroc": float("nan"), "auprc": float("nan"), "brier": brier(labels, probs),
                "prevalence": float(labels.mean()), "n": int(len(labels))}
    ci = bootstrap_ci(labels, probs, n_iter=n_boot, seed=seed)
    return {
        "auroc": float(roc_auc_score(labels, probs)),
        "auprc": float(average_precision_score(labels, probs)),
        "brier": brier(labels, probs),
        "prevalence": float(labels.mean()),
        "n": int(len(labels)),
        "auroc_ci_lo": ci["auroc"][0], "auroc_ci_hi": ci["auroc"][1],
        "auprc_ci_lo": ci["auprc"][0], "auprc_ci_hi": ci["auprc"][1],
    }
