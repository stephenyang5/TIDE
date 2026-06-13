"""Unit tests for the DeLLiriuM-matched labeling and cohort gates."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from src.labeling import (
    cam_is_positive,
    coma_first24_stays,
    label_delirium,
    prevalent_delirium_stays,
)
from src.cohort import build_cohort, exclusion_mask_vectorized


def _cam(rows):
    return pd.DataFrame(rows, columns=["stay_id", "hour_offset", "is_positive"])


def _rass(rows):
    return pd.DataFrame(rows, columns=["stay_id", "hour_offset", "rass_val"])


# ── cam_is_positive ────────────────────────────────────────────────────────────

def test_cam_is_positive_text_and_numeric():
    val = pd.Series(["Positive", "negative", "YES", "no", "unknown"])
    num = pd.Series([None, None, None, None, 1])
    res = cam_is_positive(val, num).tolist()
    assert res == [True, False, True, False, True]


# ── label: positive requires CAM+ AND assessable RASS in same 12h interval ─────

def test_label_positive_same_interval():
    # CAM+ at h30 and RASS -2 at h32 → both in interval [24,36) → positive
    cam = _cam([(1, 30.0, True)])
    rass = _rass([(1, 32.0, -2.0)])
    out = label_delirium(cam, rass, [1]).set_index("stay_id")
    assert out.loc[1, "label"] == 1
    assert out.loc[1, "first_delirium_interval_start"] == 24.0


def test_label_negative_when_rass_too_low():
    # CAM+ but only RASS = -4 (not assessable) → negative
    cam = _cam([(1, 30.0, True)])
    rass = _rass([(1, 31.0, -4.0)])
    out = label_delirium(cam, rass, [1]).set_index("stay_id")
    assert out.loc[1, "label"] == 0


def test_label_negative_when_cam_before_window():
    # CAM+ only within first 24h → not an onset event → negative (post-24h)
    cam = _cam([(1, 10.0, True)])
    rass = _rass([(1, 10.0, 0.0)])
    out = label_delirium(cam, rass, [1]).set_index("stay_id")
    assert out.loc[1, "label"] == 0


def test_label_negative_when_different_intervals():
    # CAM+ at h30 (interval0) but assessable RASS only at h50 (interval2) → negative
    cam = _cam([(1, 30.0, True)])
    rass = _rass([(1, 50.0, -1.0)])
    out = label_delirium(cam, rass, [1]).set_index("stay_id")
    assert out.loc[1, "label"] == 0


def test_label_first_onset_interval():
    # Confirmed intervals at [36,48) and [48,60); first onset start = 36
    cam = _cam([(1, 40.0, True), (1, 52.0, True)])
    rass = _rass([(1, 41.0, -1.0), (1, 53.0, 0.0)])
    out = label_delirium(cam, rass, [1]).set_index("stay_id")
    assert out.loc[1, "label"] == 1
    assert out.loc[1, "first_delirium_interval_start"] == 36.0


def test_label_includes_all_cohort_stays_as_zero():
    cam = _cam([(1, 30.0, True)])
    rass = _rass([(1, 31.0, -1.0)])
    out = label_delirium(cam, rass, [1, 2, 3]).set_index("stay_id")
    assert out.loc[1, "label"] == 1
    assert out.loc[2, "label"] == 0
    assert out.loc[3, "label"] == 0


# ── gates ──────────────────────────────────────────────────────────────────────

def test_prevalent_delirium_stays():
    cam = _cam([(1, 10.0, True), (2, 30.0, True), (3, 24.0, True)])
    prev = prevalent_delirium_stays(cam)
    assert prev == {1, 3}  # stay 2's CAM+ is after the window


def test_coma_persistent_vs_any():
    rass = _rass([(1, 2.0, -4.0), (1, 10.0, -5.0),   # persistently comatose
                  (2, 2.0, -4.0), (2, 10.0, 0.0)])   # arouses later
    assert coma_first24_stays(rass, policy="persistent") == {1}
    assert coma_first24_stays(rass, policy="any") == {1, 2}


# ── cohort ICD exclusion + age/LOS ─────────────────────────────────────────────

def test_exclusion_mask_matches_dementia_tbi_psych_not_f05():
    codes = pd.Series(["F03", "S06.9", "F20", "F99", "F05", "I10"])
    res = exclusion_mask_vectorized(codes).tolist()
    assert res == [True, True, True, True, False, False]


def test_build_cohort_applies_age_los_first_icu():
    icustays = pd.DataFrame({
        "subject_id": [1, 1, 2, 3],
        "hadm_id":    [10, 11, 20, 30],
        "stay_id":    [100, 101, 200, 300],
        "intime":  ["2150-01-01", "2150-02-01", "2150-01-01", "2150-01-01"],
        "outtime": ["2150-01-03", "2150-02-03", "2150-01-01 12:00", "2150-01-05"],
    })
    admissions = pd.DataFrame({
        "hadm_id": [10, 11, 20, 30],
        "admittime": ["2150-01-01"] * 4,
        "deathtime": [None, None, None, None],
    })
    patients = pd.DataFrame({
        "subject_id": [1, 2, 3],
        "anchor_age": [40, 10, 65],     # subject 2 is a minor
        "anchor_year": [2150, 2150, 2150],
    })
    out = build_cohort(icustays, admissions, patients)
    # subject 2 excluded (age<18 AND LOS<24); subject 1 keeps only first ICU stay
    assert set(out["stay_id"]) == {100, 300}


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
    print("All labeling/cohort tests passed.")
