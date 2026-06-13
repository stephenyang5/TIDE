"""Unit tests for baseline aggregate features and metrics."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from src.baselines import build_aggregate_matrix
from src.metrics import reliability_curve, summarize


def test_build_aggregate_matrix_values():
    feats = pd.DataFrame({
        "stay_id":     [1, 1, 1, 2],
        "hour_offset": [0, 2, 5, 0],
        "feature_name": ["heart_rate", "heart_rate", "heart_rate", "heart_rate"],
        "value":       [60.0, 80.0, 100.0, 70.0],
    })
    X = build_aggregate_matrix(feats, [1, 2], feature_names=["heart_rate"], max_hours=24)
    assert X.loc[1, "heart_rate__count"] == 3
    assert X.loc[1, "heart_rate__mean"] == 80.0
    assert X.loc[1, "heart_rate__min"] == 60.0
    assert X.loc[1, "heart_rate__max"] == 100.0
    assert X.loc[1, "heart_rate__last"] == 100.0   # latest hour_offset
    assert X.loc[2, "heart_rate__count"] == 1


def test_build_aggregate_matrix_missing_and_window():
    feats = pd.DataFrame({
        "stay_id":     [1, 1],
        "hour_offset": [0, 30],   # second obs is outside the 24h window
        "feature_name": ["spo2", "spo2"],
        "value":       [95.0, 88.0],
    })
    X = build_aggregate_matrix(feats, [1, 2], feature_names=["spo2"], max_hours=24)
    assert X.loc[1, "spo2__count"] == 1        # only the in-window obs
    assert X.loc[1, "spo2__last"] == 95.0
    # stay 2 has no data → count 0, value aggregates NaN
    assert X.loc[2, "spo2__count"] == 0
    assert np.isnan(X.loc[2, "spo2__mean"])


def test_build_aggregate_matrix_row_order_matches_request():
    feats = pd.DataFrame({
        "stay_id": [2, 1], "hour_offset": [0, 0],
        "feature_name": ["map", "map"], "value": [1.0, 2.0],
    })
    X = build_aggregate_matrix(feats, [1, 2], feature_names=["map"])
    assert list(X.index) == [1, 2]  # preserves requested order


def test_summarize_and_reliability():
    rng = np.random.default_rng(0)
    labels = np.array([0, 0, 0, 1, 1, 0, 1, 0, 1, 0])
    probs = np.array([0.1, 0.2, 0.3, 0.9, 0.8, 0.05, 0.7, 0.4, 0.6, 0.15])
    m = summarize(labels, probs, n_boot=50)
    assert 0.5 <= m["auroc"] <= 1.0
    assert 0.0 <= m["brier"] <= 1.0
    assert m["n"] == 10
    rc = reliability_curve(labels, probs, n_bins=5)
    assert len(rc["mean_pred"]) == len(rc["frac_pos"]) == len(rc["count"])
    assert rc["count"].sum() == 10


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn(); print(f"PASS {fn.__name__}")
        except Exception:
            failed += 1; print(f"FAIL {fn.__name__}"); traceback.print_exc()
    if failed:
        raise SystemExit(f"{failed} test(s) failed")
    print("All baseline tests passed.")
