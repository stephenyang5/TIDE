"""Classical baselines (Logistic Regression, HistGradientBoosting) for delirium.

Answers the question the deep model cannot on its own: *does T-PatchGNN beat
simple models on the same data?* Uses the **same test split** as ``src.train``
(stratified, seed 42) and the same feature-exclusion options, so numbers are
directly comparable.

Features: per-stay aggregates over the first ``max_hours`` of each variable's
*observed* values (from the pre-LOCF table) — count, mean, min, max, last.

Run:
    python -m src.baselines --config conservative
    python -m src.baselines --config full
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.feature_vocab import FEATURE_NAMES
from src.metrics import reliability_curve, summarize

_AGGS = ["count", "mean", "min", "max", "last"]

# Feature exclusion presets (mirror the deep-model configs)
EXCLUDE_PRESETS: dict[str, list[str]] = {
    "full": [],
    "no_cam_rass": ["cam_icu", "rass"],
    "conservative": ["cam_icu", "rass", "gcs_eye", "gcs_verbal", "gcs_motor"],
}


def build_aggregate_matrix(
    feats: pd.DataFrame,
    stay_ids: list[int],
    *,
    feature_names: list[str] | None = None,
    max_hours: int | None = 24,
) -> pd.DataFrame:
    """Per-stay aggregate feature matrix (rows = stay_ids in given order).

    Columns are ``{feature}__{agg}`` for agg in count/mean/min/max/last.
    Missing (stay, feature) combinations are NaN.
    """
    names = feature_names or FEATURE_NAMES
    df = feats[feats["feature_name"].isin(names)].copy()
    if max_hours is not None:
        df = df[df["hour_offset"] < max_hours]

    if df.empty:
        cols = [f"{f}__{a}" for f in names for a in _AGGS]
        return pd.DataFrame(0.0, index=pd.Index(stay_ids, name="stay_id"), columns=cols)

    grp = df.groupby(["stay_id", "feature_name"])["value"]
    agg = grp.agg(["count", "mean", "min", "max"])
    last = (
        df.sort_values(["stay_id", "feature_name", "hour_offset"])
        .groupby(["stay_id", "feature_name"])["value"].last()
    )
    agg["last"] = last

    wide = agg.unstack("feature_name")  # columns: MultiIndex (agg, feature)
    wide.columns = [f"{feat}__{a}" for a, feat in wide.columns]

    # Fixed column order; reindex rows to requested stay order
    cols = [f"{f}__{a}" for f in names for a in _AGGS]
    wide = wide.reindex(columns=cols)
    wide = wide.reindex(index=stay_ids)
    wide.index.name = "stay_id"
    # count columns: missing → 0 observations; value aggregates stay NaN
    count_cols = [c for c in cols if c.endswith("__count")]
    wide[count_cols] = wide[count_cols].fillna(0.0)
    return wide


def _split_indices(labels: np.ndarray, *, test_frac: float, seed: int):
    """Replicate src.train's stratified test split exactly."""
    indices = np.arange(len(labels))
    idx_trainval, idx_test = train_test_split(
        indices, test_size=test_frac, stratify=labels, random_state=seed
    )
    return idx_trainval, idx_test


def _make_models(seed: int):
    lr = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)),
    ])
    # HistGBT handles NaN natively; no imputation/scaling needed.
    hgb = HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=400, max_depth=None,
        l2_regularization=1.0, class_weight="balanced", random_state=seed,
    )
    return {"logreg": lr, "hist_gbt": hgb}


def run_baselines(
    cohort: pd.DataFrame,
    feats: pd.DataFrame,
    *,
    exclude: list[str],
    max_hours: int = 24,
    test_frac: float = 0.10,
    seed: int = 42,
    n_boot: int = 200,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, dict]]:
    """Train LR + HistGBT and evaluate on the held-out test split.

    Returns ``(summary_df, predictions, reliability)``.
    """
    labels = cohort["label"].to_numpy().astype(int)
    stay_ids = cohort["stay_id"].astype(int).tolist()
    keep = [f for f in FEATURE_NAMES if f not in set(exclude)]

    X = build_aggregate_matrix(feats, stay_ids, feature_names=keep, max_hours=max_hours)
    X_arr = X.to_numpy(dtype=float)

    idx_trainval, idx_test = _split_indices(labels, test_frac=test_frac, seed=seed)
    y_tr, y_te = labels[idx_trainval], labels[idx_test]
    X_tr, X_te = X_arr[idx_trainval], X_arr[idx_test]

    rows, preds, reliab = [], {}, {}
    for name, model in _make_models(seed).items():
        model.fit(X_tr, y_tr)
        p_te = model.predict_proba(X_te)[:, 1]
        m = summarize(y_te, p_te, n_boot=n_boot, seed=seed)
        m["model"] = name
        rows.append(m)
        preds[name] = p_te
        reliab[name] = reliability_curve(y_te, p_te)

    summary = pd.DataFrame(rows).set_index("model")
    cols = ["n", "prevalence", "auroc", "auroc_ci_lo", "auroc_ci_hi",
            "auprc", "auprc_ci_lo", "auprc_ci_hi", "brier"]
    summary = summary[[c for c in cols if c in summary.columns]]
    preds["_labels"] = y_te
    preds["_stay_ids"] = np.array(stay_ids)[idx_test]
    return summary, preds, reliab


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Classical baselines for delirium prediction.")
    p.add_argument("--cohort", type=Path, default=Path("cohort.csv"))
    p.add_argument("--features", type=Path, default=Path("features_hourly_prelocf.csv"),
                   help="Pre-LOCF (observed) long features; aggregates use true observations.")
    p.add_argument("--config", choices=list(EXCLUDE_PRESETS), default="conservative")
    p.add_argument("--exclude-features", nargs="*", default=None,
                   help="Override the preset exclusion list.")
    p.add_argument("--max-hours", type=int, default=24)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-boot", type=int, default=200)
    p.add_argument("--output-dir", type=Path, default=Path("results/baselines"))
    args = p.parse_args(argv)

    exclude = args.exclude_features if args.exclude_features is not None else EXCLUDE_PRESETS[args.config]
    print(f"Config '{args.config}' — excluding {exclude or '(none)'}")

    cohort = pd.read_csv(args.cohort)
    feats = pd.read_csv(args.features)
    print(f"Cohort: {len(cohort):,} stays  prevalence {cohort['label'].mean():.4f}")

    summary, preds, _ = run_baselines(
        cohort, feats, exclude=exclude, max_hours=args.max_hours,
        seed=args.seed, n_boot=args.n_boot,
    )
    print("\n=== Baseline test-set metrics ===")
    print(summary.round(4).to_string())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / f"baselines_{args.config}.csv"
    summary.round(6).to_csv(out)
    pred_df = pd.DataFrame({
        "stay_id": preds["_stay_ids"], "label": preds["_labels"],
        **{f"prob_{k}": v for k, v in preds.items() if not k.startswith("_")},
    })
    pred_df.to_csv(args.output_dir / f"baseline_predictions_{args.config}.csv", index=False)
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
