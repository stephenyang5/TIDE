"""Canonical 57 channel names — order matches ``01_cohort_extraction.ipynb`` CHART/LAB/DRUG dicts."""

from __future__ import annotations

# Chart (16) + Lab (24) + Drug (17) — insertion order preserved (Py 3.7+).
FEATURE_NAMES: list[str] = [
    # Chart
    "cam_icu",
    "rass",
    "gcs_eye",
    "gcs_verbal",
    "gcs_motor",
    "heart_rate",
    "sbp",
    "dbp",
    "map",
    "spo2",
    "resp_rate",
    "temperature",
    "fio2",
    "peep",
    "tidal_volume",      # itemids 224685 (observed) + 224686 (spontaneous) combined by hourly mean
    "urine_sp_gravity",  # itemids 227471 (active) + 220799 (legacy) combined by hourly mean
    # Labs
    "lactate",
    "bun",
    "creatinine",
    "glucose",
    "sodium",
    "potassium",
    "chloride",
    "bicarbonate",
    "anion_gap",
    "calcium",
    "magnesium",
    "phosphate",
    "wbc",
    "hemoglobin",
    "hematocrit",
    "platelets",
    "albumin",
    "total_bilirubin",
    "alt",
    "ast",
    "inr",
    "ptt",
    "ph",
    "nt_probnp",    # itemid 50963 — top DeLLiriuM SHAP predictor; NT-proBNP is MIMIC-IV's form
    # Drugs
    "drug_propofol",
    "drug_dexmedetomidine",
    "drug_ketamine",
    "drug_lorazepam",
    "drug_midazolam",
    "drug_fentanyl",
    "drug_fentanyl_conc",
    "drug_morphine",
    "drug_hydromorphone",
    "drug_methadone",
    "drug_norepinephrine",
    "drug_epinephrine",
    "drug_dopamine",
    "drug_vasopressin",
    "drug_phenylephrine",
    "drug_dobutamine",
    "drug_cisatracurium",
]

NUM_FEATURES = len(FEATURE_NAMES)
assert NUM_FEATURES == 57, NUM_FEATURES

_NAME_TO_IDX = {n: i for i, n in enumerate(FEATURE_NAMES)}
NAME_TO_IDX = _NAME_TO_IDX  # public alias for vectorized dataset operations

# Feature group slices (chart 0–15, labs 16–39, drugs 40–56)
CHART_FEATURES = FEATURE_NAMES[:16]
LAB_FEATURES   = FEATURE_NAMES[16:40]
DRUG_FEATURES  = FEATURE_NAMES[40:]
FEATURE_GROUPS = {"chart": CHART_FEATURES, "lab": LAB_FEATURES, "drug": DRUG_FEATURES}


def feature_to_index(name: str) -> int:
    return _NAME_TO_IDX[name]
