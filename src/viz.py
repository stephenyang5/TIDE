"""Visualisation utilities for ICU delirium prediction results.

All plot functions:
  - Accept explicit data arguments (no global state).
  - Save a PNG to *output_path* (parent dirs created automatically).
  - Optionally display inline when show=True (e.g. inside Jupyter).
  - Are safe for headless/Slurm use (Agg backend forced before pyplot import).

"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg") # force non-interactive backend before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.data.feature_vocab import CHART_FEATURES, DRUG_FEATURES, FEATURE_NAMES, LAB_FEATURES

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    pass  # older matplotlib without v0_8 aliases — fall back to default

_FIG_DPI = 150

RESULTS_DIR = Path("results")


# internal helpers

def _save(fig: plt.Figure, output_path: Path | str, show: bool) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=_FIG_DPI, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def _bootstrap_roc_band(
    labels: np.ndarray,
    probs: np.ndarray,
    fpr_grid: np.ndarray,
    n_iter: int = 200,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (tpr_lo, tpr_hi) interpolated on *fpr_grid* via bootstrap."""
    rng = np.random.default_rng(seed)
    n = len(labels)
    tpr_curves: list[np.ndarray] = []
    for _ in range(n_iter):
        idx = rng.integers(0, n, size=n)
        y_b, p_b = labels[idx], probs[idx]
        if y_b.sum() == 0 or y_b.sum() == n:
            continue
        fpr_b, tpr_b, _ = roc_curve(y_b, p_b)
        tpr_curves.append(np.interp(fpr_grid, fpr_b, tpr_b))
    if not tpr_curves:
        return np.zeros_like(fpr_grid), np.ones_like(fpr_grid)
    mat = np.vstack(tpr_curves)
    return (np.percentile(mat, 100 * alpha / 2, axis=0),
            np.percentile(mat, 100 * (1 - alpha / 2), axis=0))


# Training curves

def plot_training_curves(
    history_csv: str | Path,
    output_path: str | Path = RESULTS_DIR / "training_curves.png",
    *,
    show: bool = False,
) -> None:
    """Two-panel figure: train loss + val metrics (top) and LR trace (bottom).

    Parameters
    ----------
    history_csv
        Path to training_history.csv written by src.train.
        Expected columns: epoch, train_loss, val_auroc, val_auprc, lr.
    """
    df = pd.read_csv(history_csv)
    required = {"epoch", "train_loss", "val_auroc", "val_auprc", "lr"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"history_csv missing columns: {missing}")

    epochs = df["epoch"].to_numpy()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})

    # Top panel — loss (left y) and AUROC/AUPRC (right y)
    c_loss = "#e07b39"
    ax1.plot(epochs, df["train_loss"], color=c_loss, lw=2, label="Train loss")
    ax1.set_ylabel("BCE Loss", color=c_loss)
    ax1.tick_params(axis="y", labelcolor=c_loss)

    ax1r = ax1.twinx()
    ax1r.plot(epochs, df["val_auroc"], color="#3a7ebf", lw=2, label="Val AUROC")
    ax1r.plot(epochs, df["val_auprc"], color="#3abf7e", lw=2,
              linestyle="--", label="Val AUPRC")
    ax1r.set_ylabel("Metric")
    ax1r.set_ylim(0, 1.05)

    # Best epoch marker
    best_ep = int(df.loc[df["val_auroc"].idxmax(), "epoch"])
    best_au = float(df["val_auroc"].max())
    ax1r.axvline(best_ep, color="grey", lw=1, linestyle=":")
    ax1r.annotate(
        f"best\n{best_au:.4f}",
        xy=(best_ep, best_au), xytext=(best_ep + 0.5, best_au - 0.05),
        fontsize=8, color="grey",
    )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1r.get_legend_handles_labels()
    ax1r.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=9)
    ax1.set_title("Training Curves")

    # Bottom panel — LR
    ax2.semilogy(epochs, df["lr"], color="#9b59b6", lw=2)
    ax2.set_ylabel("Learning Rate")
    ax2.set_xlabel("Epoch")

    fig.tight_layout()
    _save(fig, output_path, show)


# ROC curve with bootstrap CI
def plot_roc_curve(
    labels: np.ndarray,
    probs: np.ndarray,
    *,
    auroc_ci: Optional[tuple[float, float]] = None,
    bootstrap_n: int = 200,
    output_path: str | Path = RESULTS_DIR / "roc_curve.png",
    show: bool = False,
) -> None:
    """ROC curve with bootstrap CI band.

    Parameters
    ----------
    auroc_ci
        Pre-computed (lo, hi) tuple from :func: src.train.bootstrap_ci.
        If None the CI band is computed internally.
    bootstrap_n
        Number of bootstrap iterations when auroc_ci is None.
    """
    fpr, tpr, _ = roc_curve(labels, probs)
    auroc_val = roc_auc_score(labels, probs)

    fpr_grid = np.linspace(0, 1, 300)
    tpr_lo, tpr_hi = _bootstrap_roc_band(labels, probs, fpr_grid, n_iter=bootstrap_n)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color="#3a7ebf", lw=2,
            label=f"AUROC = {auroc_val:.4f}")
    ax.fill_between(fpr_grid, tpr_lo, tpr_hi, alpha=0.2, color="#3a7ebf",
                    label="95 % CI (bootstrap)")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance (0.5)")

    if auroc_ci is not None:
        ax.set_title(
            f"ROC Curve — AUROC {auroc_val:.4f}"
            f"[{auroc_ci[0]:.4f}, {auroc_ci[1]:.4f}]"
        )
    else:
        ax.set_title(f"ROC Curve — AUROC {auroc_val:.4f}")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    _save(fig, output_path, show)


# Precision-Recall curve

def plot_pr_curve(
    labels: np.ndarray,
    probs: np.ndarray,
    *,
    auprc_ci: Optional[tuple[float, float]] = None,
    prevalence: Optional[float] = None,
    output_path: str | Path = RESULTS_DIR / "pr_curve.png",
    show: bool = False,
) -> None:
    """Precision-Recall curve with optional CI and random-classifier baseline.

    Parameters
    ----------
    auprc_ci
        Pre-computed (lo, hi) from bootstrap_ci.
    prevalence
        If provided, draws a horizontal dashed line at this y-level (the
        random-classifier PR baseline when all predictions equal prevalence).
    """
    precision, recall, _ = precision_recall_curve(labels, probs)
    auprc_val = average_precision_score(labels, probs)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(recall, precision, color="#e07b39", lw=2,
            label=f"AUPRC = {auprc_val:.4f}")

    if prevalence is not None:
        ax.axhline(prevalence, color="grey", linestyle="--", lw=1,
                   label=f"No-skill baseline ({prevalence:.3f})")

    if auprc_ci is not None:
        ax.set_title(
            f"Precision-Recall Curve — AUPRC {auprc_val:.4f}  "
            f"[{auprc_ci[0]:.4f}, {auprc_ci[1]:.4f}]"
        )
    else:
        ax.set_title(f"Precision-Recall Curve — AUPRC {auprc_val:.4f}")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    _save(fig, output_path, show)


# Calibration

def plot_calibration(
    labels: np.ndarray,
    probs: np.ndarray,
    *,
    n_bins: int = 10,
    strategy: str = "quantile",
    output_path: str | Path = RESULTS_DIR / "calibration.png",
    show: bool = False,
) -> None:
    """Reliability diagram (calibration plot).

    strategy="quantile" ensures equal sample counts per bin
    for imbalanced datasets (10 % positive rate means uniform bins mostly
    contain only negatives).

    NaN bins (no positives/negatives) are silently dropped.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frac_pos, mean_pred = calibration_curve(
            labels, probs, n_bins=n_bins, strategy=strategy
        )

    valid = ~(np.isnan(frac_pos) | np.isnan(mean_pred))
    frac_pos, mean_pred = frac_pos[valid], mean_pred[valid]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax.plot(mean_pred, frac_pos, "o-", color="#3abf7e", lw=2, ms=7,
            label="Model")

    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction Positive")
    ax.set_title(f"Calibration Plot ({strategy} bins, n_bins={n_bins})")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    _save(fig, output_path, show)


# 5. Score distribution

def plot_score_distribution(
    labels: np.ndarray,
    probs: np.ndarray,
    *,
    output_path: str | Path = RESULTS_DIR / "score_distribution.png",
    show: bool = False,
) -> None:
    """Violin plot of predicted probabilities split by true label."""
    df = pd.DataFrame({
        "prob":  probs,
        "label": pd.Categorical(
            ["Delirium" if l else "No Delirium" for l in labels],
            categories=["No Delirium", "Delirium"],
            ordered=True,
        ),
    })

    fig, ax = plt.subplots(figsize=(6, 6))
    palette = {"No Delirium": "#3a7ebf", "Delirium": "#e07b39"}
    sns.violinplot(
        data=df, x="label", y="prob", hue="label",
        palette=palette, inner="box", density_norm="width",
        ax=ax, legend=False,
    )
    sns.stripplot(
        data=df, x="label", y="prob", hue="label",
        palette=palette, alpha=0.08, size=2, jitter=True,
        ax=ax, legend=False,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Predicted Probability")
    ax.set_ylim(-0.02, 1.02)

    n_neg = int((labels == 0).sum())
    n_pos = int((labels == 1).sum())
    ax.set_xticklabels([f"No Delirium\n(n={n_neg:,})", f"Delirium\n(n={n_pos:,})"])
    ax.set_title("Score Distribution by True Label")
    fig.tight_layout()
    _save(fig, output_path, show)


# 6. Confusion matrix at threshold
def plot_confusion_at_threshold(
    labels: np.ndarray,
    probs: np.ndarray,
    *,
    threshold: float = 0.5,
    output_path: str | Path = RESULTS_DIR / "confusion_matrix.png",
    show: bool = False,
) -> dict[str, float]:
    """Confusion matrix heatmap at *threshold*.

    Returns a dict with sensitivity, specificity, PPV (precision), NPV, and F1.
    Row-normalised values are shown in the heatmap; raw counts are annotated.
    """
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(labels, preds)  # [[TN, FP], [FN, TP]]

    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    ppv = tp / max(tp + fp, 1)
    npv = tn / max(tn + fn, 1)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    metrics = dict(sensitivity=sensitivity, specificity=specificity,
                        ppv=ppv, npv=npv, f1=f1,
                        tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn))

    # Row-normalised for colourmap
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    class_names = ["No Delirium", "Delirium"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(
        f"Confusion Matrix  (threshold={threshold:.3f})\n"
        f"Sensitivity={sensitivity:.3f}  Specificity={specificity:.3f}  "
        f"F1={f1:.3f}"
    )

    for i in range(2):
        for j in range(2):
            raw = cm[i, j]
            norm = cm_norm[i, j]
            text_color = "white" if norm > 0.6 else "black"
            ax.text(j, i, f"{raw}\n({norm:.2f})",
                    ha="center", va="center", fontsize=11, color=text_color)

    fig.tight_layout()
    _save(fig, output_path, show)
    return metrics


# Convenience wrapper

def make_all_plots(
    predictions_csv: str | Path,
    history_csv: str | Path,
    *,
    bootstrap_iters: int = 200,
    threshold: float = 0.5,
    output_dir: str | Path = RESULTS_DIR,
    show: bool = False,
) -> None:
    """Load result CSVs and produce all six diagnostic plots.

    Parameters
    ----------
    predictions_csv
        results/test_predictions.csv from src.train
        (columns: stay_id, label, prob).
    history_csv
        results/training_history.csv from src.train
        (columns: epoch, train_loss, val_auroc, val_auprc, lr).
    bootstrap_iters
        Iterations for bootstrap CI band on the ROC curve.
    threshold
        Classification threshold for the confusion matrix.
    output_dir
        Directory where all PNGs are saved.
    show
        Call plt.show() after each figure (for interactive use).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    preds = pd.read_csv(predictions_csv)
    labels = preds["label"].to_numpy(dtype=int)
    probs = preds["prob"].to_numpy(dtype=float)
    prevalence = float(labels.mean())

    # Validate
    if np.isnan(probs).any():
        raise ValueError("NaN found in predictions — check evaluate() in train.py")
    if probs.min() < 0 or probs.max() > 1:
        raise ValueError("Predictions outside [0, 1] — ensure sigmoid was applied")

    try:
        plot_training_curves(
            history_csv,
            output_path=output_dir / "training_curves.png",
            show=show,
        )
    except FileNotFoundError:
        print(f"WARNING: history_csv not found ({history_csv}) — skipping training curves")
    except Exception as exc:
        print(f"WARNING: could not plot training curves: {exc}")

    plot_roc_curve(
        labels, probs,
        bootstrap_n=bootstrap_iters,
        output_path=output_dir / "roc_curve.png",
        show=show,
    )
    plot_pr_curve(
        labels, probs,
        prevalence=prevalence,
        output_path=output_dir / "pr_curve.png",
        show=show,
    )
    plot_calibration(
        labels, probs,
        output_path=output_dir / "calibration.png",
        show=show,
    )
    plot_score_distribution(
        labels, probs,
        output_path=output_dir / "score_distribution.png",
        show=show,
    )

    # Confusion at default threshold
    metrics_default = plot_confusion_at_threshold(
        labels, probs, threshold=threshold,
        output_path=output_dir / f"confusion_at_{threshold:.2f}.png",
        show=show,
    )

    # Confusion at optimal threshold
    fpr, tpr, thresholds = roc_curve(labels, probs)
    j_idx = int(np.argmax(tpr - fpr))
    opt_thr = float(thresholds[j_idx])
    metrics_opt = plot_confusion_at_threshold(
        labels, probs, threshold=opt_thr,
        output_path=output_dir / f"confusion_optimal_{opt_thr:.3f}.png",
        show=show,
    )

    print(f"\nAll plots saved to {output_dir}/")
    print(f"Default threshold ({threshold}): "
          f"sensitivity={metrics_default['sensitivity']:.3f}  "
          f"specificity={metrics_default['specificity']:.3f}  "
          f"F1={metrics_default['f1']:.3f}")
    print(f"Optimal threshold ({opt_thr:.3f}): "
          f"sensitivity={metrics_opt['sensitivity']:.3f}  "
          f"specificity={metrics_opt['specificity']:.3f}  "
          f"F1={metrics_opt['f1']:.3f}")
