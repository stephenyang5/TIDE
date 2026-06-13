"""Canonical 54 channel names — order matches ``01_cohort_extraction.ipynb`` CHART/LAB/DRUG dicts."""
# Chart (14) + Lab (23) + Drug (17) — insertion order preserved (Py 3.7+).
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
assert NUM_FEATURES == 54, NUM_FEATURES

_NAME_TO_IDX = {n: i for i, n in enumerate(FEATURE_NAMES)}
NAME_TO_IDX = _NAME_TO_IDX 

# Feature group slices (chart 0–13, labs 14–36, drugs 37–53)
CHART_FEATURES = FEATURE_NAMES[:14]
LAB_FEATURES   = FEATURE_NAMES[14:37]
DRUG_FEATURES  = FEATURE_NAMES[37:]
FEATURE_GROUPS = {"chart": CHART_FEATURES, "labs": LAB_FEATURES, "drugs": DRUG_FEATURES}


def feature_to_index(name: str) -> int:
    return _NAME_TO_IDX[name]
