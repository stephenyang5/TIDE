"""Feature extraction, hourly aggregation, honest LOCF, and data-coverage gates.

This module centralizes the feature-engineering logic so it is importable, testable, and
consistent with src.data.feature_vocab.

Key fixes over the original notebook implementation:

Drug itemids resolved by label against d_items at runtime
(resolve_drug_itemids) instead of hardcoded numbers. 

Honest LOCF (densify_and_locf): the hourly series is reindexed to a
complete 0..max_hours−1 grid per (stay, feature) and then
forward-filled, matching what LOCF imputation implies. The pre-densified
rows remain the faithful observation mask.

Drug aggregation: drug amounts are summed within an hour
  - cumulative dose, while chart/lab values are averaged.
Data-coverage gate (stays_with_insufficient_data) implementing the
DeLLiriuM "exclude missing data in the first 24 h" criterion.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src.data.feature_vocab import DRUG_FEATURES, FEATURE_NAMES

# Chart itemids
CHART_ITEMS: dict[int, str] = {
    228332: "cam_icu", 228096: "rass",
    220739: "gcs_eye", 223900: "gcs_verbal", 223901: "gcs_motor",
    220045: "heart_rate", 220179: "sbp", 220180: "dbp", 220052: "map",
    220277: "spo2", 220210: "resp_rate", 223761: "temperature",
    223835: "fio2", 220339: "peep",
    224685: "tidal_volume", 224686: "tidal_volume",        # observed + spontaneous
    227471: "urine_sp_gravity", 220799: "urine_sp_gravity",  # active + legacy
}

# Lab itemids
LAB_ITEMS: dict[int, str] = {
    50813: "lactate", 51006: "bun", 50912: "creatinine", 50931: "glucose",
    50983: "sodium", 50971: "potassium", 50902: "chloride", 50882: "bicarbonate",
    50868: "anion_gap", 50893: "calcium", 50960: "magnesium", 50970: "phosphate",
    51301: "wbc", 51222: "hemoglobin", 51221: "hematocrit", 51265: "platelets",
    50862: "albumin", 50885: "total_bilirubin", 50861: "alt", 50878: "ast",
    51237: "inr", 51275: "ptt", 50820: "ph", 50963: "nt_probnp",
}

# Drug label patterns 
DRUG_LABEL_PATTERNS: dict[str, tuple[str, str | None]] = {
    "drug_propofol": (r"propofol", None),
    "drug_dexmedetomidine": (r"dexmedetomidine|precedex", None),
    "drug_ketamine": (r"ketamine", None),
    "drug_lorazepam": (r"lorazepam|ativan", None),
    "drug_midazolam": (r"midazolam|versed", None),
    "drug_fentanyl": (r"fentanyl", r"conc"),
    "drug_fentanyl_conc": (r"fentanyl.*conc", None),
    "drug_morphine": (r"morphine", None),
    "drug_hydromorphone": (r"hydromorphone|dilaudid", None),
    "drug_methadone": (r"methadone", None),
    "drug_norepinephrine": (r"norepinephrine|levophed", None),
    "drug_epinephrine": (r"(?<!nor)epinephrine", None),
    "drug_dopamine": (r"dopamine", None),
    "drug_vasopressin": (r"vasopressin", None),
    "drug_phenylephrine": (r"phenylephrine|neo-?synephrine", None),
    "drug_dobutamine": (r"dobutamine", None),
    "drug_cisatracurium": (r"cisatracurium|nimbex", None),
}


# drug itemid resolution / verification

def resolve_drug_itemids(
    d_items: pd.DataFrame,
    patterns: dict[str, tuple[str, str | None]] | None = None,
) -> tuple[dict[int, str], pd.DataFrame]:
    """Resolve drug feature names to itemids by matching d_items.label.

    Parameters
    ----------
    d_items : DataFrame with at least itemid and label.

    Returns
    -------
    (itemid_to_name, report)
        itemid_to_name maps every resolved itemid to its drug feature name. 
        report has one row per drug feature: 
        feature_name, n_itemids, itemids, labels,
        status ("ok" / "NOT FOUND").
    """
    pats = patterns or DRUG_LABEL_PATTERNS
    df = d_items.copy()
    if "linksto" in df.columns:
        df = df[df["linksto"].astype(str).str.lower().eq("inputevents")]
    labels = df["label"].astype(str)

    itemid_to_name: dict[int, str] = {}
    rows: list[dict] = []
    for name, (inc, exc) in pats.items():
        m = labels.str.contains(inc, case=False, regex=True, na=False)
        if exc is not None:
            m &= ~labels.str.contains(exc, case=False, regex=True, na=False)
        hits = df[m]
        ids = [int(i) for i in hits["itemid"].tolist()]
        for i in ids:
            itemid_to_name[i] = name
        rows.append({
            "feature_name": name,
            "n_itemids": len(ids),
            "itemids": ids,
            "labels": hits["label"].astype(str).tolist(),
            "status": "ok" if ids else "NOT FOUND",
        })
    return itemid_to_name, pd.DataFrame(rows)


def verify_itemid_map(itemid_map: dict[int, str], d_items: pd.DataFrame) -> pd.DataFrame:
    """Report each itemid's actual d_items label vs. the expected feature name."""
    lut = d_items.set_index("itemid")["label"].astype(str).to_dict()
    rows = []
    for itemid, name in itemid_map.items():
        label = lut.get(itemid, "<NOT FOUND>")
        token = name.replace("drug_", "").split("_")[0].lower()
        ok = token in label.lower() if label != "<NOT FOUND>" else False
        rows.append({"itemid": itemid, "expected": name, "label": label,
                     "status": "ok" if ok else "CHECK"})
    return pd.DataFrame(rows)


# hourly aggregation

def aggregate_hourly(
    events: pd.DataFrame,
    *,
    drug_feature_names: set[str] | None = None,
) -> pd.DataFrame:
    """Aggregate long events to one value per (stay_id, hour_offset, feature_name).

    Drugs are summed (cumulative hourly dose) - all other features are
    averaged. events needs columns stay_id, hour_offset,
    feature_name, value.
    """
    drugs = set(drug_feature_names if drug_feature_names is not None else DRUG_FEATURES)
    if events.empty:
        return events.copy()
    ev = events.dropna(subset=["value"]).copy()
    ev["hour_offset"] = ev["hour_offset"].astype(int)

    is_drug = ev["feature_name"].isin(drugs)
    drug_part = (
        ev[is_drug].groupby(["stay_id", "hour_offset", "feature_name"], as_index=False)["value"].sum()
    )
    other_part = (
        ev[~is_drug].groupby(["stay_id", "hour_offset", "feature_name"], as_index=False)["value"].mean()
    )
    out = pd.concat([drug_part, other_part], ignore_index=True)
    return out.sort_values(["stay_id", "feature_name", "hour_offset"]).reset_index(drop=True)


# ── honest LOCF ──────────────────────────────────────────────────────────────────

def densify_and_locf(hourly: pd.DataFrame, *, max_hours: int = 24) -> pd.DataFrame:
    """Reindex each (stay, feature) to a full 0..max_hours-1 grid, then forward-fill.

    Hours before the first observation stay missing and are dropped. 
    Returns long format (stay_id, hour_offset, feature_name, value).
    """
    if hourly.empty:
        return hourly.copy()
    parts = []
    full_hours = np.arange(max_hours)
    for (sid, feat), g in hourly.groupby(["stay_id", "feature_name"], sort=False):
        s = (
            g.set_index("hour_offset")["value"]
            .reindex(full_hours) # densify to complete grid
            .ffill() # carry last observation forward
        )
        s = s.dropna() # drop leading hours before first obs
        if s.empty:
            continue
        parts.append(pd.DataFrame({
            "stay_id": sid, "hour_offset": s.index.astype(int),
            "feature_name": feat, "value": s.to_numpy(),
        }))
    if not parts:
        return hourly.iloc[0:0].copy()
    return pd.concat(parts, ignore_index=True)


# data-coverage

def first24_coverage(
    hourly: pd.DataFrame,
    *,
    max_hours: int = 24,
    feature_names: list[str] | None = None,
) -> pd.DataFrame:
    """Per-stay coverage in the first window: distinct features + total obs."""
    feats = set(feature_names or FEATURE_NAMES)
    sub = hourly[(hourly["hour_offset"] < max_hours) & hourly["feature_name"].isin(feats)]
    if sub.empty:
        return pd.DataFrame(columns=["stay_id", "n_distinct_features", "n_observations"])
    g = sub.groupby("stay_id")
    return pd.DataFrame({
        "n_distinct_features": g["feature_name"].nunique(),
        "n_observations": g.size(),
    }).reset_index()


def stays_with_insufficient_data(
    hourly: pd.DataFrame,
    all_stay_ids: list[int] | set[int],
    *,
    max_hours: int = 24,
    min_distinct_features: int = 5,
    min_observations: int = 10,
    feature_names: list[str] | None = None,
) -> set[int]:
    """stay_ids failing the first-window data-coverage requirement (to exclude).

    A stay is insufficient if, in the first max_hours, it has fewer than
    min_distinct_features distinct features OR fewer than min_observations
    total observations. Stays with no first-window data at all are included.
    """
    cov = first24_coverage(hourly, max_hours=max_hours, feature_names=feature_names)
    ok = set(
        cov.loc[
            (cov["n_distinct_features"] >= min_distinct_features)
            & (cov["n_observations"] >= min_observations),
            "stay_id",
        ].astype(int)
    )
    return {int(s) for s in all_stay_ids} - ok
