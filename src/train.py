"""Training script for ICU delirium onset prediction.

Architecture: T-PatchGNN backbone (DeliriumClassifier) trained with weighted
BCEWithLogitsLoss. Primary metric: AUROC. Secondary: AUPRC.

Benchmark target:
  - Best structured-EHR deep learning: AUROC ~78.1 (external validation)
  - DeLLiriuM LLM (345M params): AUROC ~82.5 (external validation)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.data.patch_dataset import ICUPatchDataset, collate_patches, compute_per_feature_minmax
from src.models.delirium_backbone import DeliriumClassifier


def bootstrap_ci(
    labels: np.ndarray,
    probs: np.ndarray,
    n_iter: int = 200,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, tuple[float, float]]:
    """Return 95% CI for AUROC and AUPRC via bootstrap resampling.

    Degenerate bootstrap samples (all-one-class) are skipped.
    Returns auroc (lo, hi), auprc (lo, hi).
    If fewer than 10 valid samples are accumulated a warning is printed and
    (nan, nan) is returned for both metrics.
    """
    rng = np.random.default_rng(seed)
    aurocs: list[float] = []
    auprcs: list[float] = []
    n = len(labels)
    for _ in range(n_iter):
        idx = rng.integers(0, n, size=n)
        y_b = labels[idx]
        p_b = probs[idx]
        if y_b.sum() == 0 or y_b.sum() == len(y_b):
            continue  # degenerate sample — skip, do not bias distribution
        aurocs.append(float(roc_auc_score(y_b, p_b)))
        auprcs.append(float(average_precision_score(y_b, p_b)))

    lo_q, hi_q = alpha / 2, 1 - alpha / 2
    if len(aurocs) < 10:
        print(f"WARNING: only {len(aurocs)} valid bootstrap samples — CI unreliable")
        nan2: tuple[float, float] = (float("nan"), float("nan"))
        return {"auroc": nan2, "auprc": nan2}

    return {
        "auroc": (float(np.quantile(aurocs, lo_q)), float(np.quantile(aurocs, hi_q))),
        "auprc": (float(np.quantile(auprcs, lo_q)), float(np.quantile(auprcs, hi_q))),
    }


# =============== Helpers ============================
def to_device(batch: dict, device: torch.device) -> dict:
    """Move all tensor values in a collate_patches batch to device."""
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float, np.ndarray, np.ndarray, list[int]]:
    """Return (AUROC, AUPRC, probs, labels, stay_ids) on loader.

    Uses sigmoid of logits for probabilities. stay_ids are collected
    directly from each batch so ordering is guaranteed correct regardless of
    DataLoader shuffle state.
    """
    model.eval()
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_sids: list[int] = []

    for batch in loader:
        batch = to_device(batch, device)
        logits = model(batch).squeeze(-1) # (B,)
        probs = torch.sigmoid(logits).cpu().numpy()
        labels = batch["label"].float().cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels)
        all_sids.extend(batch["stay_id"] if isinstance(batch["stay_id"], list)
                        else batch["stay_id"].tolist())

    probs_arr = np.concatenate(all_probs)
    labels_arr = np.concatenate(all_labels)

    if labels_arr.sum() == 0 or labels_arr.sum() == len(labels_arr):
        return 0.0, 0.0, probs_arr, labels_arr, all_sids

    auroc = float(roc_auc_score(labels_arr, probs_arr))
    auprc = float(average_precision_score(labels_arr, probs_arr))
    return auroc, auprc, probs_arr, labels_arr, all_sids


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train delirium onset classifier")
    # Data
    parser.add_argument("--cohort", type=Path, default=Path("cohort.csv"))
    parser.add_argument("--features", type=Path, default=Path("features_hourly.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--max-hours", type=int,  default=24,
                        help="Use only the first N hours of ICU stay (DeLLiriuM: 24)")
    # Split
    parser.add_argument("--val-frac", type=float, default=0.10)
    parser.add_argument("--test-frac", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    # Model
    parser.add_argument("--hid-dim", type=int, default=32)
    parser.add_argument("--n-layer", type=int, default=2,
                        help="Number of intra+inter-series blocks")
    parser.add_argument("--nhead", type=int, default=4,
                        help="Transformer attention heads")
    parser.add_argument("--tf-layer", type=int, default=2,
                        help="Transformer encoder layers per block")
    parser.add_argument("--node-dim", type=int, default=10,
                        help="Adaptive graph node embedding dim")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-patches", type=int, default=512)
    # Training
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Gradient clip max_norm; 0 = disabled")
    # LR scheduler (ReduceLROnPlateau on val AUROC)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--lr-patience", type=int, default=5,
                        help="Scheduler patience in epochs")
    parser.add_argument("--lr-min", type=float, default=1e-5)
    # Early stopping
    parser.add_argument("--patience", type=int, default=10,
                        help="Early-stopping patience (val AUROC); 0 = disabled")
    parser.add_argument("--min-delta", type=float, default=1e-4,
                        help="Minimum AUROC improvement to reset patience counter")
    # Evaluation / outputs
    parser.add_argument("--bootstrap-iters", type=int, default=200,
                        help="Bootstrap CI iterations; 0 = skip")
    parser.add_argument("--history-csv",
                        type=Path, default=Path("results/training_history.csv"))
    parser.add_argument("--predictions-csv",
                        type=Path, default=Path("results/test_predictions.csv"))
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load full cohort + features once 
    print("Loading dataset …")
    t0 = time.time()
    full_ds = ICUPatchDataset(
        cohort_path=args.cohort,
        features_path=args.features,
        max_hours=args.max_hours,
    )
    print(f"Dataset loaded in {time.time()-t0:.1f}s  ({len(full_ds):,} stays)")

    labels = full_ds.cohort["label"].to_numpy()
    stay_ids = full_ds.cohort["stay_id"].to_numpy()
    indices = np.arange(len(full_ds))

    # Stratified 80 / 10 / 10 split (matches DeLLiriuM evaluation protocol)
    idx_trainval, idx_test = train_test_split(
        indices, test_size=args.test_frac, stratify=labels, random_state=args.seed
    )
    val_frac_adjusted = args.val_frac / (1.0 - args.test_frac)
    idx_train, idx_val = train_test_split(
        idx_trainval,
        test_size=val_frac_adjusted,
        stratify=labels[idx_trainval],
        random_state=args.seed,
    )

    print(
        f"Split — train: {len(idx_train):,}  val: {len(idx_val):,}  test: {len(idx_test):,}"
    )

    # Normalisation stats from training split only
    train_stay_ids = set(stay_ids[idx_train])
    train_feats    = full_ds.feats[full_ds.feats["stay_id"].isin(train_stay_ids)]
    v_mins, v_maxs = compute_per_feature_minmax(train_feats)

    def _make_ds(ids: np.ndarray) -> ICUPatchDataset:
        sid_list  = list(stay_ids[ids])
        sid_set   = set(sid_list)
        # Pass pre-filtered features DataFrame to avoid re-reading from disk
        feats_sub = full_ds.feats[full_ds.feats["stay_id"].isin(sid_set)]
        return ICUPatchDataset(
            cohort_path=args.cohort,
            features_path=args.features,
            max_hours=args.max_hours,
            stay_ids=sid_list,
            value_mins=v_mins,
            value_maxs=v_maxs,
            features_df=feats_sub,
        )

    train_ds = _make_ds(idx_train)
    val_ds = _make_ds(idx_val)
    test_ds = _make_ds(idx_test)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_patches, num_workers=2, pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_patches, num_workers=2,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_patches, num_workers=2,
    )

    # Class-balanced loss 
    train_labels = labels[idx_train]
    n_pos = int(train_labels.sum())
    n_neg = len(train_labels) - n_pos
    print(
        f"Train labels — positive: {n_pos:,}  negative: {n_neg:,}  "
        f"prevalence: {100 * n_pos / len(train_labels):.1f}%"
    )

    pos_weight = torch.tensor(n_neg / max(n_pos, 1), dtype=torch.float32).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Model
    model = DeliriumClassifier(
        hid_dim=args.hid_dim,
        n_layer=args.n_layer,
        nhead=args.nhead,
        tf_layer=args.tf_layer,
        node_dim=args.node_dim,
        dropout=args.dropout,
        max_patches=args.max_patches,
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max", # maximise val AUROC
        factor=args.lr_factor,
        patience=args.lr_patience,
        min_lr=args.lr_min,
    )

    # Training loop
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.history_csv.parent.mkdir(parents=True, exist_ok=True)

    best_val_auroc = 0.0
    best_epoch = 0
    patience_ctr = 0
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = len(train_loader)
        log_every = max(1, n_batches // 5)   # print ~5 progress updates per epoch

        for batch_i, batch in enumerate(train_loader, 1):
            batch = to_device(batch, device)
            logits = model(batch).squeeze(-1)          # (B,)
            loss = criterion(logits, batch["label"].float())
            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total_loss += loss.item()

            if batch_i % log_every == 0 or batch_i == n_batches:
                print(
                    f"Epoch {epoch:3d}/{args.epochs} "
                    f"[{batch_i:4d}/{n_batches}]  "
                    f"batch_loss={loss.item():.4f}",
                    flush=True,
                )

        avg_loss = total_loss / max(n_batches, 1)
        val_auroc, val_auprc, _, _, _ = evaluate(model, val_loader, device)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:3d} | loss {avg_loss:.4f} "
            f"| val AUROC {val_auroc:.4f} | val AUPRC {val_auprc:.4f} "
            f"| lr {current_lr:.2e}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": avg_loss,
            "val_auroc": val_auroc,
            "val_auprc": val_auprc,
            "lr": current_lr,
        })

        # LR scheduler step (maximise val AUROC)
        scheduler.step(val_auroc)

        # Checkpoint & early stopping
        if val_auroc > best_val_auroc + args.min_delta:
            best_val_auroc = val_auroc
            best_epoch = epoch
            patience_ctr = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_auroc": val_auroc,
                    "val_auprc": val_auprc,
                    "args": vars(args),
                },
                args.output_dir / "best_model.pt",
            )
            print(f"New best checkpoint  (val AUROC {best_val_auroc:.4f})")
        else:
            patience_ctr += 1
            if args.patience > 0 and patience_ctr >= args.patience:
                print(
                    f"Early stopping at epoch {epoch} "
                    f"(no improvement for {args.patience} epochs)"
                )
                break

    # Save training history
    pd.DataFrame(history).to_csv(args.history_csv, index=False)
    print(f"\nTraining history to {args.history_csv}")

    # Final test evaluation
    print(f"\nBest checkpoint: epoch {best_epoch} — val AUROC {best_val_auroc:.4f}")
    ckpt = torch.load(args.output_dir / "best_model.pt", weights_only=False,
                      map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    test_auroc, test_auprc, test_probs, test_labels, test_sids = evaluate(
        model, test_loader, device
    )
    print(f"Test AUROC: {test_auroc:.4f}")
    print(f"Test AUPRC: {test_auprc:.4f}")

    # Bootstrap CI
    if args.bootstrap_iters > 0:
        print(f"\nBootstrap CI ({args.bootstrap_iters} iterations)")
        ci = bootstrap_ci(test_labels, test_probs, n_iter=args.bootstrap_iters,
                          seed=args.seed)
        print(f"AUROC 95% CI: [{ci['auroc'][0]:.4f}, {ci['auroc'][1]:.4f}]")
        print(f"AUPRC 95% CI: [{ci['auprc'][0]:.4f}, {ci['auprc'][1]:.4f}]")

    # Save test predictions
    args.predictions_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "stay_id": test_sids,
        "label": test_labels.astype(int),
        "prob": test_probs,
    }).to_csv(args.predictions_csv, index=False)
    print(f"Test predictions to {args.predictions_csv}")


if __name__ == "__main__":
    main()
