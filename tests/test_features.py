"""Unit tests for feature extraction, aggregation, honest LOCF, and gates."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from src.features import (
    aggregate_hourly,
    densify_and_locf,
    first24_coverage,
    resolve_drug_itemids,
    stays_with_insufficient_data,
    verify_itemid_map,
)


# ── drug itemid resolution (fixes ketamine / methadone bugs) ───────────────────

def _synthetic_d_items():
    # Includes the previously-broken cases: a real ketamine/methadone row plus
    # the wrong "Sheath Insertion" itemid that must NOT match methadone.
    return pd.DataFrame({
        "itemid": [222168, 225150, 9001, 221385, 9002, 9003, 225942,
                   9004, 9005, 225761, 221906, 221289, 221662, 222315,
                   221749, 9006, 221555, 221833],
        "label": ["Propofol", "Dexmedetomidine (Precedex)", "Ketamine",
                  "Lorazepam (Ativan)", "Midazolam (Versed)", "Fentanyl",
                  "Fentanyl (Concentrate)", "Morphine Sulfate", "Methadone",
                  "Sheath Insertion", "Norepinephrine", "Epinephrine",
                  "Dopamine", "Vasopressin", "Phenylephrine", "Dobutamine",
                  "Cisatracurium", "Hydromorphone (Dilaudid)"],
        "linksto": ["inputevents"] * 18,
    })


def test_resolve_drug_itemids_fixes_ketamine_and_methadone():
    d_items = _synthetic_d_items()
    itemid_to_name, report = resolve_drug_itemids(d_items)
    # ketamine resolves to the real Ketamine row (9001), not "NOT FOUND"
    assert itemid_to_name.get(9001) == "drug_ketamine"
    # methadone resolves to the real Methadone row (9005), NOT Sheath Insertion (225761)
    assert itemid_to_name.get(9005) == "drug_methadone"
    assert 225761 not in itemid_to_name
    rep = report.set_index("feature_name")
    assert rep.loc["drug_ketamine", "status"] == "ok"
    assert rep.loc["drug_methadone", "status"] == "ok"


def test_resolve_disambiguates_epinephrine_from_norepinephrine():
    d_items = _synthetic_d_items()
    itemid_to_name, _ = resolve_drug_itemids(d_items)
    assert itemid_to_name.get(221289) == "drug_epinephrine"     # Epinephrine
    assert itemid_to_name.get(221906) == "drug_norepinephrine"  # Norepinephrine
    # The norepinephrine row must not be tagged as epinephrine
    assert itemid_to_name.get(221906) != "drug_epinephrine"


def test_resolve_separates_fentanyl_base_and_concentrate():
    d_items = _synthetic_d_items()
    itemid_to_name, _ = resolve_drug_itemids(d_items)
    assert itemid_to_name.get(9003) == "drug_fentanyl"        # "Fentanyl"
    assert itemid_to_name.get(225942) == "drug_fentanyl_conc"  # "Fentanyl (Concentrate)"


def test_resolve_reports_not_found_when_missing():
    d_items = pd.DataFrame({"itemid": [1], "label": ["Propofol"], "linksto": ["inputevents"]})
    _, report = resolve_drug_itemids(d_items)
    rep = report.set_index("feature_name")
    assert rep.loc["drug_ketamine", "status"] == "NOT FOUND"


# ── hourly aggregation: drugs summed, others averaged ──────────────────────────

def test_aggregate_hourly_sum_vs_mean():
    events = pd.DataFrame({
        "stay_id":     [1, 1, 1, 1],
        "hour_offset": [0, 0, 0, 0],
        "feature_name": ["drug_propofol", "drug_propofol", "heart_rate", "heart_rate"],
        "value":       [10.0, 5.0, 80.0, 100.0],
    })
    out = aggregate_hourly(events).set_index("feature_name")
    assert out.loc["drug_propofol", "value"] == 15.0   # summed
    assert out.loc["heart_rate", "value"] == 90.0       # averaged


# ── honest LOCF: densify to full grid then forward-fill ────────────────────────

def test_densify_and_locf_fills_gaps():
    # heart_rate observed at hours 0 and 3 → hours 1,2 carried forward from 0
    hourly = pd.DataFrame({
        "stay_id": [1, 1],
        "hour_offset": [0, 3],
        "feature_name": ["heart_rate", "heart_rate"],
        "value": [70.0, 90.0],
    })
    out = densify_and_locf(hourly, max_hours=5).set_index("hour_offset")["value"]
    assert out.loc[0] == 70.0
    assert out.loc[1] == 70.0  # carried forward
    assert out.loc[2] == 70.0  # carried forward
    assert out.loc[3] == 90.0
    assert out.loc[4] == 90.0  # carried forward to end of window
    assert len(out) == 5


def test_densify_no_backfill_before_first_obs():
    # first observation at hour 2 → hours 0,1 remain missing (dropped)
    hourly = pd.DataFrame({
        "stay_id": [1], "hour_offset": [2], "feature_name": ["spo2"], "value": [95.0],
    })
    out = densify_and_locf(hourly, max_hours=4)
    assert set(out["hour_offset"]) == {2, 3}
    assert 0 not in set(out["hour_offset"])


# ── data-coverage gate ──────────────────────────────────────────────────────────

def test_first24_coverage_and_insufficient_gate():
    rows = []
    # stay 1: rich coverage (6 features x 2 hours = 12 obs)
    for f in ["heart_rate", "sbp", "dbp", "spo2", "resp_rate", "map"]:
        for h in [0, 1]:
            rows.append({"stay_id": 1, "hour_offset": h, "feature_name": f, "value": 1.0})
    # stay 2: sparse (2 features, 2 obs)
    rows.append({"stay_id": 2, "hour_offset": 0, "feature_name": "heart_rate", "value": 1.0})
    rows.append({"stay_id": 2, "hour_offset": 0, "feature_name": "sbp", "value": 1.0})
    hourly = pd.DataFrame(rows)

    cov = first24_coverage(hourly, max_hours=24).set_index("stay_id")
    assert cov.loc[1, "n_distinct_features"] == 6
    assert cov.loc[2, "n_distinct_features"] == 2

    insufficient = stays_with_insufficient_data(
        hourly, [1, 2, 3], min_distinct_features=5, min_observations=10
    )
    assert insufficient == {2, 3}  # stay 2 sparse, stay 3 has no data


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
    print("All feature tests passed.")
