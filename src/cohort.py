"""Build the ICU stay cohort with DeLLiriuM-compatible inclusion/exclusion gates.

This module owns the demographic / administrative cohort only:

  * first ICU stay per patient
  * age >= 18
  * ICU LOS >= min_los_hours
  * exclude death within exclude_early_death_hours of ICU admission
  * exclude pre-existing dementia (ICD F03), TBI (S06), and chronic psychiatric
    diagnoses (F20–F99, excluding F05 delirium itself)

The delirium label is not assigned here — it comes from CAM-ICU + RASS
chart events via src.labeling. This is a
deliberate change: the previous ICD-based delirium_icd label conflicted with
the CAM/RASS label used for training.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.mimic_paths import diagnoses_icd_path

# Pre-existing cognitive / psychiatric confounds to exclude (ICD-10-CM prefixes,
# punctuation stripped). F05 (delirium) is intentionally NOT excluded.
#   F03  — unspecified dementia
#   S06  — intracranial / traumatic brain injury
#   F20–F99 (except F05) — chronic psychiatric disorders
EXCLUSION_ICD10_PREFIXES: tuple[str, ...] = tuple(
    ["F03", "S06"] + [f"F{n:02d}" for n in range(20, 100) if n != 5]
)


def _normalize_icd_series(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.upper()
        .str.replace(".", "", regex=False)
        .str.strip()
        .replace({"NAN": ""})
    )


def exclusion_mask_vectorized(icd_code: pd.Series) -> pd.Series:
    """Boolean Series: True where an ICD row is a dementia/TBI/psych confound.

    Prefix matching is applied to normalized codes. ICD-9 equivalents are not
    matched (documented limitation); MIMIC-IV is overwhelmingly ICD-10 for the
    relevant period.
    """
    n = _normalize_icd_series(icd_code)
    return n.str.startswith(EXCLUSION_ICD10_PREFIXES)


def load_exclusion_hadm_ids(
    diag_path: Path | None = None,
    chunksize: int = 500_000,
) -> set[int]:
    """hadm_ids carrying any dementia/TBI/chronic-psych diagnosis."""
    path = diag_path or diagnoses_icd_path()
    excl: set[int] = set()
    for chunk in pd.read_csv(
        path,
        compression="infer",
        chunksize=chunksize,
        usecols=["hadm_id", "icd_code"],
    ):
        m = exclusion_mask_vectorized(chunk["icd_code"])
        if m.any():
            excl.update(chunk.loc[m, "hadm_id"].dropna().astype(int).tolist())
    return excl


def build_cohort(
    icustays: pd.DataFrame,
    admissions: pd.DataFrame,
    patients: pd.DataFrame,
    exclusion_hadm_ids: set[int] | None = None,
    *,
    min_los_hours: float = 24.0,
    min_age: int = 18,
    first_icu_only: bool = True,
    exclude_early_death_hours: float = 48.0,
) -> pd.DataFrame:
    """Build the demographic ICU cohort (no delirium label)."""
    icu = icustays.copy()
    icu["intime"] = pd.to_datetime(icu["intime"], errors="coerce")
    icu["outtime"] = pd.to_datetime(icu["outtime"], errors="coerce")
    icu["los_hours"] = (icu["outtime"] - icu["intime"]).dt.total_seconds() / 3600.0

    # ICU LOS >= threshold
    icu = icu[icu["los_hours"] >= min_los_hours].copy()

    adm_cols = [
        "hadm_id", "admittime", "dischtime", "deathtime", "admission_type",
        "admission_location", "discharge_location", "insurance", "language",
        "marital_status", "race", "hospital_expire_flag",
    ]
    adm_keep = [c for c in adm_cols if c in admissions.columns]
    adm = admissions[adm_keep].drop_duplicates(subset=["hadm_id"])
    out = icu.merge(adm, on="hadm_id", how="left")

    pat_cols = [
        c for c in ["subject_id", "gender", "anchor_age", "anchor_year", "dod"]
        if c in patients.columns
    ]
    pat = patients[pat_cols].drop_duplicates(subset=["subject_id"])
    out = out.merge(pat, on="subject_id", how="left")

    adm_year = pd.to_datetime(out["admittime"], errors="coerce").dt.year
    out["age_at_admission"] = (
        out["anchor_age"] + (adm_year - out["anchor_year"])
    ).clip(upper=91)

    # Adults only >= min age
    age_basis = out["age_at_admission"].fillna(out.get("anchor_age"))
    out = out[age_basis >= min_age].copy()

    # First ICU admission per patient
    if first_icu_only:
        out = out.sort_values("intime").groupby("subject_id", as_index=False).first()

    # Exclude early in-hospital deaths
    if exclude_early_death_hours > 0 and "deathtime" in out.columns:
        deathtime = pd.to_datetime(out["deathtime"], errors="coerce")
        hours_to_death = (deathtime - out["intime"]).dt.total_seconds() / 3600.0
        early_death = deathtime.notna() & (hours_to_death < exclude_early_death_hours)
        out = out[~early_death].copy()

    # Exclude dementia / TBI / chronic-psych confounds by hadm_id
    if exclusion_hadm_ids:
        out = out[~out["hadm_id"].isin(exclusion_hadm_ids)].copy()

    return out.reset_index(drop=True)
