"""End-to-end regeneration of ``cohort.csv`` + ``features_hourly*.csv``.

This is the single runnable entry point that wires the *tested* building blocks
(:mod:`src.cohort`, :mod:`src.features`, :mod:`src.labeling`) into the full
pipeline, replacing the buggy inline logic in ``01_cohort_extraction.ipynb``.

Pipeline
--------
1. Build the unlabeled demographic cohort (``src.cohort.build_cohort``).
2. Scan ``chartevents`` / ``labevents`` / ``inputevents`` (chunked) for the
   feature itemids, resolving drug itemids by label (``src.features``).
3. Aggregate hourly (drugs summed, others averaged) → pre-LOCF table.
4. Label with CAM-ICU + RASS over 12 h windows, applying prevalent-delirium,
   coma, missing-data, and negative-censoring gates (``src.labeling``).
5. Apply honest LOCF (``src.features.densify_and_locf``).
6. Write ``cohort.csv``, ``features_hourly.csv``, ``features_hourly_prelocf.csv``.

The IO/scan layer (:func:`main`) needs MIMIC-IV on Oscar. The assembly logic
(:func:`assemble_dataset`) is a pure function tested on synthetic data.

Run:
    python -m src.build_features --min-los-hours 24 --max-hours 24
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.cohort import build_cohort, load_exclusion_hadm_ids
from src.data.feature_vocab import DRUG_FEATURES, FEATURE_NAMES
from src.features import (
    CHART_ITEMS,
    LAB_ITEMS,
    aggregate_hourly,
    densify_and_locf,
    resolve_drug_itemids,
)
from src.labeling import (
    CAM_ICU_ITEMID,
    RASS_ITEMID,
    assessed_after_window,
    cam_is_positive,
    coma_first24_stays,
    label_delirium,
    prevalent_delirium_stays,
)
from src.mimic_paths import (
    admissions_path,
    diagnoses_icd_path,
    icustays_path,
    patients_path,
    resolve_table,
    icu_dir,
    hosp_dir,
)

COHORT_OUTPUT_COLUMNS = [
    "stay_id", "subject_id", "hadm_id", "label", "first_delirium_interval_start",
    "los_hours", "age_at_admission", "anchor_age", "gender", "race",
    "insurance", "marital_status", "first_careunit",
]


def assemble_dataset(
    cohort: pd.DataFrame,
    feature_events: pd.DataFrame,
    cam_events: pd.DataFrame,
    rass_events: pd.DataFrame,
    *,
    max_hours: int = 24,
    min_distinct_features: int = 5,
    min_observations: int = 10,
    coma_policy: str = "persistent",
    require_post_window_assessment: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Pure assembly: apply gates, label, aggregate, and LOCF.

    Parameters
    ----------
    cohort : demographic cohort (from ``build_cohort``); must have ``stay_id``.
    feature_events : long ``stay_id, hour_offset, feature_name, value`` for the
        first ``max_hours`` (all 57 features incl. cam_icu/rass within window).
    cam_events : full-stay ``stay_id, hour_offset, is_positive`` (for labeling).
    rass_events : full-stay ``stay_id, hour_offset, rass_val`` (for labeling).

    Returns
    -------
    (cohort_labeled, hourly_prelocf, hourly_locf, report)
    """
    from src.features import stays_with_insufficient_data

    report: dict = {"n_start": int(len(cohort))}
    cohort = cohort.copy()
    cohort_ids = set(cohort["stay_id"].astype(int))

    # --- Gate 1: prevalent delirium (CAM+ in first window) ---
    prevalent = prevalent_delirium_stays(cam_events, prediction_window_hours=max_hours)
    cohort = cohort[~cohort["stay_id"].isin(prevalent)]
    report["n_after_prevalent_excl"] = int(len(cohort))

    # --- Gate 2: coma in first window (never arousable) ---
    comatose = coma_first24_stays(
        rass_events, prediction_window_hours=max_hours, policy=coma_policy
    )
    cohort = cohort[~cohort["stay_id"].isin(comatose)]
    report["n_after_coma_excl"] = int(len(cohort))

    cohort_ids = set(cohort["stay_id"].astype(int))
    fe = feature_events[feature_events["stay_id"].isin(cohort_ids)].copy()
    fe = fe[(fe["hour_offset"] >= 0) & (fe["hour_offset"] < max_hours)]

    # --- Hourly aggregation (drugs summed, others averaged) = pre-LOCF table ---
    hourly_prelocf = aggregate_hourly(fe, drug_feature_names=set(DRUG_FEATURES))

    # --- Gate 3: missing data in first window ---
    insufficient = stays_with_insufficient_data(
        hourly_prelocf, cohort_ids, max_hours=max_hours,
        min_distinct_features=min_distinct_features,
        min_observations=min_observations, feature_names=FEATURE_NAMES,
    )
    cohort = cohort[~cohort["stay_id"].isin(insufficient)]
    report["n_after_missing_data_excl"] = int(len(cohort))

    cohort_ids = set(cohort["stay_id"].astype(int))

    # --- Label (CAM+ & RASS>=-3 in same 12h post-window interval) ---
    labels = label_delirium(cam_events, rass_events, cohort_ids,
                            prediction_window_hours=max_hours)

    # --- Gate 4: negative-class censoring (require post-window assessment) ---
    if require_post_window_assessment:
        assessed = assessed_after_window(cam_events, rass_events,
                                         prediction_window_hours=max_hours)
        neg = labels["label"] == 0
        keep = labels["label"] == 1
        keep = keep | (neg & labels["stay_id"].isin(assessed))
        labels = labels[keep]
        cohort_ids = set(labels["stay_id"].astype(int))
        cohort = cohort[cohort["stay_id"].isin(cohort_ids)]
    report["n_after_negative_censoring"] = int(len(cohort))

    # --- Merge labels into cohort ---
    cohort = cohort.merge(labels, on="stay_id", how="inner")
    report["n_positive"] = int(cohort["label"].sum())
    report["prevalence"] = float(cohort["label"].mean()) if len(cohort) else float("nan")

    # --- Restrict feature tables to the final labeled cohort ---
    final_ids = set(cohort["stay_id"].astype(int))
    hourly_prelocf = hourly_prelocf[hourly_prelocf["stay_id"].isin(final_ids)].reset_index(drop=True)
    hourly_locf = densify_and_locf(hourly_prelocf, max_hours=max_hours)

    # --- Order cohort columns (keep whatever exists) ---
    cols = [c for c in COHORT_OUTPUT_COLUMNS if c in cohort.columns]
    extra = [c for c in cohort.columns if c not in cols]
    cohort = cohort[cols + extra]
    return cohort.reset_index(drop=True), hourly_prelocf, hourly_locf, report


# ── MIMIC scan helpers ─────────────────────────────────────────────────────────

def _hour_offset(charttime: pd.Series, intime: pd.Series) -> pd.Series:
    return ((charttime - intime).dt.total_seconds() / 3600.0)


def _scan_chartevents(path: Path, cohort: pd.DataFrame, chunksize: int):
    """Return (feature_events_first_window-ready, cam_events, rass_events).

    Chart feature rows are kept full-stay here; the window restriction happens
    in ``assemble_dataset``. CAM/RASS are also returned separately for labeling.
    """
    keep_items = set(CHART_ITEMS) | {CAM_ICU_ITEMID, RASS_ITEMID}
    stay_ids = set(cohort["stay_id"].astype(int))
    intime_map = cohort.set_index("stay_id")["intime"]

    feat_rows, cam_rows, rass_rows = [], [], []
    for chunk in pd.read_csv(
        path, usecols=["stay_id", "itemid", "charttime", "value", "valuenum"],
        chunksize=chunksize, low_memory=False, compression="infer",
    ):
        sub = chunk[chunk["itemid"].isin(keep_items) & chunk["stay_id"].isin(stay_ids)]
        if sub.empty:
            continue
        sub = sub.copy()
        sub["charttime"] = pd.to_datetime(sub["charttime"], errors="coerce")
        sub["intime"] = pd.to_datetime(sub["stay_id"].map(intime_map), errors="coerce")
        sub = sub.dropna(subset=["charttime", "intime"])
        sub["hour_offset"] = _hour_offset(sub["charttime"], sub["intime"])

        cam = sub[sub["itemid"] == CAM_ICU_ITEMID]
        if not cam.empty:
            cam_rows.append(pd.DataFrame({
                "stay_id": cam["stay_id"].astype(int),
                "hour_offset": cam["hour_offset"],
                "is_positive": cam_is_positive(cam["value"], cam["valuenum"]),
            }))
        rass = sub[sub["itemid"] == RASS_ITEMID]
        if not rass.empty:
            rass_rows.append(pd.DataFrame({
                "stay_id": rass["stay_id"].astype(int),
                "hour_offset": rass["hour_offset"],
                "rass_val": pd.to_numeric(rass["valuenum"], errors="coerce"),
            }).dropna(subset=["rass_val"]))

        # Feature rows: numeric value; cam_icu uses binary positive flag
        feat = sub[sub["itemid"].isin(CHART_ITEMS)].copy()
        feat["value_num"] = pd.to_numeric(feat["valuenum"], errors="coerce")
        cam_mask = feat["itemid"] == CAM_ICU_ITEMID
        if cam_mask.any():
            feat.loc[cam_mask, "value_num"] = cam_is_positive(
                feat.loc[cam_mask, "value"], feat.loc[cam_mask, "valuenum"]
            ).astype(float)
        feat["feature_name"] = feat["itemid"].map(CHART_ITEMS)
        feat = feat.rename(columns={"value_num": "fval"})
        feat = feat[["stay_id", "hour_offset", "feature_name", "fval"]].dropna(subset=["fval"])
        feat = feat.rename(columns={"fval": "value"})
        feat["stay_id"] = feat["stay_id"].astype(int)
        feat_rows.append(feat)

    def _cat(rows, cols):
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=cols)

    feature_events = _cat(feat_rows, ["stay_id", "hour_offset", "feature_name", "value"])
    cam_events = _cat(cam_rows, ["stay_id", "hour_offset", "is_positive"])
    rass_events = _cat(rass_rows, ["stay_id", "hour_offset", "rass_val"])
    return feature_events, cam_events, rass_events


def _scan_labevents(path: Path, cohort: pd.DataFrame, chunksize: int) -> pd.DataFrame:
    keep_items = set(LAB_ITEMS)
    hadm_ids = set(cohort["hadm_id"].astype("Int64").dropna().astype(int))
    stay_hadm = cohort[["stay_id", "hadm_id", "intime", "outtime"]].copy()
    rows = []
    for chunk in pd.read_csv(
        path, usecols=["hadm_id", "itemid", "charttime", "valuenum"],
        chunksize=chunksize, low_memory=False, compression="infer",
    ):
        sub = chunk[chunk["itemid"].isin(keep_items) & chunk["hadm_id"].isin(hadm_ids)]
        if not sub.empty:
            rows.append(sub.copy())
    if not rows:
        return pd.DataFrame(columns=["stay_id", "hour_offset", "feature_name", "value"])
    lab = pd.concat(rows, ignore_index=True)
    lab["charttime"] = pd.to_datetime(lab["charttime"], errors="coerce")
    lab = lab.merge(stay_hadm, on="hadm_id", how="inner")
    lab["intime"] = pd.to_datetime(lab["intime"], errors="coerce")
    lab["outtime"] = pd.to_datetime(lab["outtime"], errors="coerce")
    lab = lab[(lab["charttime"] >= lab["intime"]) & (lab["charttime"] <= lab["outtime"])]
    lab["hour_offset"] = _hour_offset(lab["charttime"], lab["intime"])
    lab["feature_name"] = lab["itemid"].map(LAB_ITEMS)
    lab = lab.rename(columns={"valuenum": "value"})
    lab["stay_id"] = lab["stay_id"].astype(int)
    return lab[["stay_id", "hour_offset", "feature_name", "value"]].dropna(subset=["value"])


def _scan_inputevents(path: Path, cohort: pd.DataFrame, itemid_to_name: dict[int, str],
                      chunksize: int) -> pd.DataFrame:
    keep_items = set(itemid_to_name)
    stay_ids = set(cohort["stay_id"].astype(int))
    intime_map = cohort.set_index("stay_id")["intime"]
    rows = []
    for chunk in pd.read_csv(
        path, usecols=["stay_id", "itemid", "starttime", "amount"],
        chunksize=chunksize, low_memory=False, compression="infer",
    ):
        sub = chunk[chunk["itemid"].isin(keep_items) & chunk["stay_id"].isin(stay_ids)]
        if not sub.empty:
            rows.append(sub.copy())
    if not rows:
        return pd.DataFrame(columns=["stay_id", "hour_offset", "feature_name", "value"])
    drug = pd.concat(rows, ignore_index=True)
    drug["charttime"] = pd.to_datetime(drug["starttime"], errors="coerce")
    drug["intime"] = pd.to_datetime(drug["stay_id"].map(intime_map), errors="coerce")
    drug = drug.dropna(subset=["charttime", "intime"])
    drug["hour_offset"] = _hour_offset(drug["charttime"], drug["intime"])
    drug["feature_name"] = drug["itemid"].map(itemid_to_name)
    drug["value"] = pd.to_numeric(drug["amount"], errors="coerce")
    drug["stay_id"] = drug["stay_id"].astype(int)
    return drug[["stay_id", "hour_offset", "feature_name", "value"]].dropna(subset=["value"])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Regenerate cohort + hourly features.")
    p.add_argument("--mimic-root", type=Path, default=None)
    p.add_argument("--min-los-hours", type=float, default=24.0)
    p.add_argument("--min-age", type=int, default=18)
    p.add_argument("--max-hours", type=int, default=24)
    p.add_argument("--min-distinct-features", type=int, default=5)
    p.add_argument("--min-observations", type=int, default=10)
    p.add_argument("--coma-policy", choices=["persistent", "any"], default="persistent")
    p.add_argument("--keep-unassessed-negatives", action="store_true",
                   help="Disable negative-class censoring (sensitivity analysis).")
    p.add_argument("--chunksize", type=int, default=500_000)
    p.add_argument("--out-cohort", type=Path, default=Path("cohort.csv"))
    p.add_argument("--out-features", type=Path, default=Path("features_hourly.csv"))
    p.add_argument("--out-prelocf", type=Path, default=Path("features_hourly_prelocf.csv"))
    args = p.parse_args(argv)

    if args.mimic_root is not None:
        import os
        os.environ["MIMIC_ROOT"] = str(args.mimic_root)

    icu_path = icustays_path()
    for path, label in ((icu_path, "icustays"), (admissions_path(), "admissions"),
                        (patients_path(), "patients"), (diagnoses_icd_path(), "diagnoses_icd")):
        if not path.is_file():
            print(f"Missing {label}: {path}", file=sys.stderr)
            return 1

    print("Loading base tables...")
    icustays = pd.read_csv(icu_path, compression="infer")
    admissions = pd.read_csv(admissions_path(), compression="infer")
    patients = pd.read_csv(patients_path(), compression="infer")
    d_items = pd.read_csv(resolve_table(icu_dir(), "d_items"), compression="infer")

    print("Scanning diagnoses_icd for dementia/TBI/psych exclusions...")
    exclusion_ids = load_exclusion_hadm_ids()

    cohort = build_cohort(icustays, admissions, patients,
                          exclusion_hadm_ids=exclusion_ids,
                          min_los_hours=args.min_los_hours, min_age=args.min_age)
    cohort["intime"] = pd.to_datetime(cohort["intime"], errors="coerce")
    cohort["outtime"] = pd.to_datetime(cohort["outtime"], errors="coerce")
    print(f"  Demographic cohort: {len(cohort):,} stays")

    print("Resolving drug itemids by label...")
    itemid_to_name, drug_report = resolve_drug_itemids(d_items)
    print(drug_report[["feature_name", "n_itemids", "status"]].to_string(index=False))

    print("Scanning chartevents...")
    chart_feat, cam_events, rass_events = _scan_chartevents(
        resolve_table(icu_dir(), "chartevents"), cohort, args.chunksize)
    print(f"  chart feature rows: {len(chart_feat):,}  cam: {len(cam_events):,}  rass: {len(rass_events):,}")

    print("Scanning labevents...")
    lab_feat = _scan_labevents(resolve_table(hosp_dir(), "labevents"), cohort, args.chunksize)
    print(f"  lab feature rows: {len(lab_feat):,}")

    print("Scanning inputevents...")
    drug_feat = _scan_inputevents(resolve_table(icu_dir(), "inputevents"),
                                  cohort, itemid_to_name, args.chunksize)
    print(f"  drug feature rows: {len(drug_feat):,}")

    feature_events = pd.concat([chart_feat, lab_feat, drug_feat], ignore_index=True)

    print("Assembling (gates + label + LOCF)...")
    cohort_out, prelocf, locf, report = assemble_dataset(
        cohort, feature_events, cam_events, rass_events,
        max_hours=args.max_hours,
        min_distinct_features=args.min_distinct_features,
        min_observations=args.min_observations,
        coma_policy=args.coma_policy,
        require_post_window_assessment=not args.keep_unassessed_negatives,
    )

    print("\nFunnel:")
    for k, v in report.items():
        print(f"  {k:30s}: {v}")

    cohort_out.to_csv(args.out_cohort, index=False)
    prelocf.to_csv(args.out_prelocf, index=False)
    locf.to_csv(args.out_features, index=False)
    print(f"\nWrote {args.out_cohort} ({len(cohort_out):,} stays), "
          f"{args.out_features} ({len(locf):,} rows), "
          f"{args.out_prelocf} ({len(prelocf):,} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
