"""Interpretability evaluation for the trained delirium classifier.

Loads a saved checkpoint, reconstructs the *exact* train/val/test split
used during training (same seed + fractions), then runs the full
interpretability suite on the held-out test set:

  1. Feature & patch ablation (AUROC drop)
  2. Adaptive graph A_p extraction and visualisation
  3. Transformer attention-weight capture
  4. Patient embedding t-SNE / UMAP
  5. Integrated Gradients attribution heatmaps  (slow; skip with --skip-ig)

If the checkpoint was trained with ``--exclude-features``, the same channels
are zeroed (values + point_mask) on every forward pass, matching ``src.train``
evaluation. Use ``--exclude-features`` to override or set exclusions when the
checkpoint has no saved ``exclude_features`` (e.g. older runs).

All outputs are saved to ``results/interpret/`` by default.

Run from project root:
  python -m src.interpret_eval --checkpoint checkpoints/best_model.pt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.data.patch_dataset import ICUPatchDataset, collate_patches, compute_per_feature_minmax
from src.data.feature_vocab import NAME_TO_IDX
from src.models.delirium_backbone import DeliriumClassifier
from src.interpret.feature_ablation import (
    baseline_auroc,
    run_feature_ablation,
    run_patch_ablation,
    plot_feature_importance,
)
from src.interpret.graph_viz import (
    extract_adjacency,
    extract_patient_adjacency,
    plot_adjacency,
    plot_patient_graph_examples,
    summarize_graph_heterogeneity,
)
from src.interpret.attention import aggregate_attention, plot_attention
from src.interpret.embed_viz import extract_embeddings, plot_tsne
from src.interpret.integrated_gradients import aggregate_ig, plot_ig_heatmap, plot_ig_value_vs_mask
from src.interpret.permutation_importance import run_permutation_importance, plot_permutation_importance


def _resolve_exclude_feature_idxs(
    ckpt_args: dict,
    cli_names: list[str] | None,
) -> list[int]:
    """Feature indices to mask (must match training) — CLI overrides checkpoint."""
    names = cli_names if cli_names is not None else (ckpt_args.get("exclude_features") or [])
    idxs: list[int] = []
    for name in names:
        if name not in NAME_TO_IDX:
            raise ValueError(
                f"Unknown feature name '{name}'. Valid names are in src.data.feature_vocab."
            )
        idxs.append(NAME_TO_IDX[name])
    return idxs


# ── helpers ──────────────────────────────────────────────────────────────────

def _to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def _build_test_loader(
    ckpt_args: dict,
    cohort_path: Path,
    features_path: Path,
    batch_size: int,
    seed: int,
) -> DataLoader:
    """Reconstruct the held-out test split used during training."""
    print("Loading dataset …")
    full_ds = ICUPatchDataset(
        cohort_path=cohort_path,
        features_path=features_path,
        max_hours=ckpt_args.get("max_hours", 24),
    )

    labels   = full_ds.cohort["label"].to_numpy()
    stay_ids = full_ds.cohort["stay_id"].to_numpy()
    indices  = np.arange(len(full_ds))

    test_frac = ckpt_args.get("test_frac", 0.10)
    val_frac  = ckpt_args.get("val_frac", 0.10)

    idx_trainval, idx_test = train_test_split(
        indices, test_size=test_frac, stratify=labels, random_state=seed
    )
    val_frac_adj = val_frac / (1.0 - test_frac)
    idx_train, _ = train_test_split(
        idx_trainval,
        test_size=val_frac_adj,
        stratify=labels[idx_trainval],
        random_state=seed,
    )

    # Normalisation from train split only
    train_sids  = set(stay_ids[idx_train])
    train_feats = full_ds.feats[full_ds.feats["stay_id"].isin(train_sids)]
    v_mins, v_maxs = compute_per_feature_minmax(train_feats)

    test_sid_list = list(stay_ids[idx_test])
    test_sid_set  = set(test_sid_list)
    feats_test    = full_ds.feats[full_ds.feats["stay_id"].isin(test_sid_set)]

    test_ds = ICUPatchDataset(
        cohort_path=cohort_path,
        features_path=features_path,
        max_hours=ckpt_args.get("max_hours", 24),
        stay_ids=test_sid_list,
        value_mins=v_mins,
        value_maxs=v_maxs,
        features_df=feats_test,
        cohort_df=full_ds.cohort,
    )

    print(f"Test set: {len(test_ds):,} stays")
    return DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_patches,
        num_workers=2,
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run interpretability suite on the trained delirium model."
    )
    parser.add_argument("--checkpoint",  type=Path, default=Path("checkpoints/best_model.pt"))
    parser.add_argument("--cohort",      type=Path, default=Path("cohort.csv"))
    parser.add_argument("--features",    type=Path, default=Path("features_hourly.csv"))
    parser.add_argument("--output-dir",  type=Path, default=Path("results/interpret"))
    parser.add_argument("--batch-size",  type=int,  default=32)
    parser.add_argument("--ig-steps",   type=int,  default=50,
                        help="Integrated gradient interpolation steps (default 50)")
    parser.add_argument("--seed",        type=int,  default=42)
    parser.add_argument("--val-frac",    type=float, default=0.10,
                        help="Must match value used during training (default 0.10)")
    parser.add_argument("--test-frac",   type=float, default=0.10,
                        help="Must match value used during training (default 0.10)")
    parser.add_argument("--skip-ig",     action="store_true",
                        help="Skip Integrated Gradients (slowest step)")
    parser.add_argument("--skip-perm",   action="store_true",
                        help="Skip permutation importance (slow)")
    parser.add_argument("--perm-repeats", type=int, default=10,
                        help="Permutation repeats per feature (default 10)")
    parser.add_argument(
        "--exclude-features",
        nargs="*",
        default=None,
        metavar="NAME",
        help="Feature names to mask (same as training --exclude-features). "
             "If omitted, uses exclude_features from the checkpoint args.",
    )
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load checkpoint ───────────────────────────────────────────────────
    print(f"Loading checkpoint from {args.checkpoint} …")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ckpt_args: dict = ckpt.get("args", {})

    # Override split fracs with CLI values if not in checkpoint
    ckpt_args.setdefault("val_frac",  args.val_frac)
    ckpt_args.setdefault("test_frac", args.test_frac)
    ckpt_args.setdefault("seed",      args.seed)

    exclude_idxs = _resolve_exclude_feature_idxs(ckpt_args, args.exclude_features)
    if exclude_idxs:
        src = "CLI" if args.exclude_features is not None else "checkpoint"
        names = args.exclude_features if args.exclude_features is not None else (
            ckpt_args.get("exclude_features") or []
        )
        print(f"Masking excluded features ({src}): {list(names)} → indices {exclude_idxs}")

    # ── Build model ───────────────────────────────────────────────────────
    model = DeliriumClassifier(
        hid_dim    = ckpt_args.get("hid_dim",     32),
        n_layer    = ckpt_args.get("n_layer",      2),
        nhead      = ckpt_args.get("nhead",        4),
        tf_layer   = ckpt_args.get("tf_layer",     2),
        node_dim   = ckpt_args.get("node_dim",    10),
        dropout    = ckpt_args.get("dropout",     0.1),
        max_patches= ckpt_args.get("max_patches", 512),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"Model loaded (val AUROC at checkpoint: {ckpt.get('val_auroc', 'N/A')})")

    # ── Reconstruct test split ────────────────────────────────────────────
    test_loader = _build_test_loader(
        ckpt_args=ckpt_args,
        cohort_path=args.cohort,
        features_path=args.features,
        batch_size=args.batch_size,
        seed=ckpt_args["seed"],
    )

    summary_rows: list[dict] = []

    # ── 1. Feature & patch ablation ───────────────────────────────────────
    print("\n[1/7] Feature & patch ablation …")
    t0 = time.time()
    ref = baseline_auroc(model, test_loader, device, exclude_idxs=exclude_idxs)
    print(f"  Baseline AUROC = {ref:.4f}")
    df_feat  = run_feature_ablation(
        model, test_loader, device, ref_auroc=ref, exclude_idxs=exclude_idxs
    )
    df_patch = run_patch_ablation(
        model, test_loader, device, ref_auroc=ref, exclude_idxs=exclude_idxs
    )

    feat_csv = args.output_dir / "feature_ablation.csv"
    patch_csv = args.output_dir / "patch_ablation.csv"
    df_feat.to_csv(feat_csv,  index=False)
    df_patch.to_csv(patch_csv, index=False)
    print(f"  Saved {feat_csv}\n  Saved {patch_csv}")

    plot_feature_importance(df_feat, df_patch, args.output_dir)

    top5 = df_feat.head(5)["feature_name"].tolist()
    print(f"  Top-5 features by AUROC drop: {top5}")
    summary_rows.append({"step": "Feature ablation", "elapsed_s": time.time() - t0,
                          "note": f"top5={top5}"})

    # ── 2. Permutation importance (headline) ──────────────────────────────
    if args.skip_perm:
        print("\n[2/7] Permutation importance — SKIPPED (--skip-perm)")
        summary_rows.append({"step": "Permutation importance", "elapsed_s": 0, "note": "skipped"})
    else:
        print(f"\n[2/7] Permutation importance ({args.perm_repeats} repeats/feature) …")
        t0 = time.time()
        df_perm = run_permutation_importance(
            model, test_loader, device,
            ref_auroc=ref, exclude_idxs=exclude_idxs, n_repeats=args.perm_repeats,
        )
        perm_csv = args.output_dir / "permutation_importance.csv"
        df_perm.to_csv(perm_csv, index=False)
        plot_permutation_importance(df_perm, args.output_dir)
        top5p = df_perm.head(5)["feature_name"].tolist()
        print(f"  Top-5 permutation: {top5p}")
        summary_rows.append({"step": "Permutation importance", "elapsed_s": time.time() - t0,
                              "note": f"top5={top5p}"})

    # ── 3. Adaptive graph (cohort mean) ─────────────────────────────────
    print("\n[3/7] Adaptive graph extraction …")
    t0 = time.time()
    adp_dict = extract_adjacency(model, test_loader, device, exclude_idxs=exclude_idxs)
    np.save(args.output_dir / "graph_adj.npy",
            np.stack([adp_dict["pos"], adp_dict["neg"], adp_dict["all"]], axis=0))
    plot_adjacency(adp_dict, args.output_dir)
    summary_rows.append({"step": "Graph A_p", "elapsed_s": time.time() - t0, "note": ""})

    # ── 4. Per-patient graph heterogeneity ────────────────────────────────
    print("\n[4/7] Per-patient graph heterogeneity …")
    t0 = time.time()
    pt_graphs, pt_labels, pt_probs = extract_patient_adjacency(
        model, test_loader, device, exclude_idxs=exclude_idxs
    )
    np.savez(
        args.output_dir / "graph_per_patient.npz",
        graphs=pt_graphs, labels=pt_labels, probs=pt_probs,
    )
    summarize_graph_heterogeneity(pt_graphs, args.output_dir, layer=-1)
    plot_patient_graph_examples(pt_graphs, pt_labels, pt_probs, args.output_dir, layer=-1)
    summary_rows.append({"step": "Graph heterogeneity", "elapsed_s": time.time() - t0,
                          "note": f"N={len(pt_graphs)}"})

    # ── 5. Transformer attention ──────────────────────────────────────────
    print("\n[5/7] Transformer attention extraction …")
    t0 = time.time()
    attn_mean = aggregate_attention(model, test_loader, device, exclude_idxs=exclude_idxs)
    if attn_mean is not None:
        np.save(args.output_dir / "attention.npy", attn_mean)
        plot_attention(attn_mean, args.output_dir)
        note = f"shape={attn_mean.shape}"
    else:
        note = "skipped (flash attention active — no weights returned)"
        print(f"  {note}")
    summary_rows.append({"step": "Attention", "elapsed_s": time.time() - t0, "note": note})

    # ── 6. Patient embeddings ─────────────────────────────────────────────
    print("\n[6/7] Patient embedding projection …")
    t0 = time.time()
    embeddings, emb_labels, emb_probs = extract_embeddings(
        model, test_loader, device, exclude_idxs=exclude_idxs
    )
    np.save(args.output_dir / "embeddings.npy",        embeddings)
    np.save(args.output_dir / "embeddings_labels.npy", emb_labels)
    np.save(args.output_dir / "embeddings_probs.npy",  emb_probs)
    plot_tsne(embeddings, emb_labels, emb_probs, args.output_dir)
    summary_rows.append({"step": "Embeddings", "elapsed_s": time.time() - t0,
                          "note": f"N={len(embeddings)}"})

    # ── 7. Integrated Gradients (values + mask) ───────────────────────────
    if args.skip_ig:
        print("\n[7/7] Integrated Gradients — SKIPPED (--skip-ig)")
        summary_rows.append({"step": "IG", "elapsed_s": 0, "note": "skipped"})
    else:
        print(f"\n[7/7] Integrated Gradients ({args.ig_steps} steps, values + mask) …")
        t0 = time.time()
        attrs, attrs_mask, ig_labels = aggregate_ig(
            model, test_loader, n_steps=args.ig_steps, device=device,
            exclude_idxs=exclude_idxs, include_mask=True,
        )
        np.save(args.output_dir / "ig_attrs.npy", attrs)
        np.save(args.output_dir / "ig_attrs_mask.npy", attrs_mask)
        np.save(args.output_dir / "ig_labels.npy", ig_labels)
        plot_ig_heatmap(attrs, ig_labels, args.output_dir)
        plot_ig_value_vs_mask(attrs, attrs_mask, args.output_dir)
        summary_rows.append({"step": "IG", "elapsed_s": time.time() - t0,
                              "note": f"steps={args.ig_steps}, values+mask"})

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Interpretability run complete.")
    print(f"Outputs saved to: {args.output_dir.resolve()}")
    print()
    df_sum = pd.DataFrame(summary_rows)
    print(df_sum.to_string(index=False))
    df_sum.to_csv(args.output_dir / "summary.csv", index=False)

    # Print ablation table
    print("\nTop-15 features by AUROC drop (ablation):")
    print(df_feat.head(15).to_string(index=False))

    if not args.skip_perm:
        print("\nTop-15 features by permutation importance:")
        print(df_perm.head(15).to_string(index=False))

    print("\nPatch ablation:")
    print(df_patch.to_string(index=False))


if __name__ == "__main__":
    main()
