"""Smoke tests for the pure assembly core of src.build_features."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from src.build_features import assemble_dataset


def _coverage_rows(stay_id, hours=(0, 1)):
    """6 distinct features × given hours → passes the missing-data gate."""
    rows = []
    for f in ["heart_rate", "sbp", "dbp", "spo2", "resp_rate", "map"]:
        for h in hours:
            rows.append({"stay_id": stay_id, "hour_offset": h, "feature_name": f, "value": 1.0})
    return rows


def _build_inputs():
    # 5 stays exercising every gate:
    #   1 positive | 2 prevalent | 3 coma | 4 negative-assessed | 5 negative-unassessed
    cohort = pd.DataFrame({"stay_id": [1, 2, 3, 4, 5], "los_hours": [100.0] * 5})

    feats = []
    for sid in [1, 2, 4, 5]:            # stay 3 is comatose; coverage irrelevant
        feats += _coverage_rows(sid)
    feature_events = pd.DataFrame(feats)

    cam_events = pd.DataFrame([
        {"stay_id": 1, "hour_offset": 30.0, "is_positive": True},    # onset
        {"stay_id": 2, "hour_offset": 10.0, "is_positive": True},    # prevalent (<=24)
        {"stay_id": 4, "hour_offset": 30.0, "is_positive": False},   # assessed, negative
    ])
    rass_events = pd.DataFrame([
        {"stay_id": 1, "hour_offset": 31.0, "rass_val": -2.0},       # assessable w/ CAM+
        {"stay_id": 3, "hour_offset": 2.0,  "rass_val": -5.0},       # coma in first 24h
        {"stay_id": 3, "hour_offset": 10.0, "rass_val": -4.0},
    ])
    return cohort, feature_events, cam_events, rass_events


def test_assemble_applies_all_gates_and_labels():
    cohort, fe, cam, rass = _build_inputs()
    out, prelocf, locf, report = assemble_dataset(
        cohort, fe, cam, rass, max_hours=24,
        min_distinct_features=5, min_observations=10,
        require_post_window_assessment=True,
    )
    # stay 2 (prevalent), 3 (coma), 5 (unassessed negative) all removed
    assert set(out["stay_id"]) == {1, 4}
    assert out.set_index("stay_id").loc[1, "label"] == 1
    assert out.set_index("stay_id").loc[4, "label"] == 0
    assert report["n_positive"] == 1
    assert abs(report["prevalence"] - 0.5) < 1e-9
    # feature tables restricted to the final cohort
    assert set(prelocf["stay_id"]) <= {1, 4}
    assert set(locf["stay_id"]) <= {1, 4}


def test_negative_censoring_sensitivity_keeps_unassessed():
    cohort, fe, cam, rass = _build_inputs()
    out, *_ = assemble_dataset(
        cohort, fe, cam, rass, max_hours=24,
        min_distinct_features=5, min_observations=10,
        require_post_window_assessment=False,   # disable censoring
    )
    # stay 5 (unassessed negative) is retained when censoring is off
    assert set(out["stay_id"]) == {1, 4, 5}


def test_locf_densifies_relative_to_prelocf():
    cohort, fe, cam, rass = _build_inputs()
    _, prelocf, locf, _ = assemble_dataset(cohort, fe, cam, rass, max_hours=24)
    # LOCF carries values forward to the window end → more rows than pre-LOCF
    assert len(locf) >= len(prelocf)


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    if failed:
        raise SystemExit(f"{failed} test(s) failed")
    print("All build_features tests passed.")
