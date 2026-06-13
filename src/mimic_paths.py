"""Resolved paths to MIMIC-IV modules on Oscar.

Override the filesystem root with the ``MIMIC_ROOT`` environment variable
(default: ``/oscar/data/shared/ursa/mimic-iv``).
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_MIMIC_ROOT = Path("/oscar/data/shared/ursa/mimic-iv")
ICU_VERSION = "3.1"
HOSP_VERSION = "3.1"
ED_VERSION = "2.2"
NOTE_VERSION = "2.2"


def mimic_root() -> Path:
    return Path(os.environ.get("MIMIC_ROOT", str(DEFAULT_MIMIC_ROOT))).expanduser().resolve()


def icu_dir() -> Path:
    return mimic_root() / "icu" / ICU_VERSION


def hosp_dir() -> Path:
    return mimic_root() / "hosp" / HOSP_VERSION


def ed_dir() -> Path:
    return mimic_root() / "ed" / ED_VERSION


def note_dir() -> Path:
    return mimic_root() / "note" / NOTE_VERSION


def icustays_path() -> Path:
    return icu_dir() / "icustays.csv"


def admissions_path() -> Path:
    return hosp_dir() / "admissions.csv"


def patients_path() -> Path:
    return hosp_dir() / "patients.csv"


def diagnoses_icd_path() -> Path:
    return hosp_dir() / "diagnoses_icd.csv"


def d_icd_diagnoses_path() -> Path:
    return hosp_dir() / "d_icd_diagnoses.csv"


def resolve_table(module_dir: Path, stem: str) -> Path:
    """Return ``module_dir / f'{stem}.csv'`` or ``…/f'{stem}.csv.gz'`` if it exists."""
    for name in (f"{stem}.csv", f"{stem}.csv.gz"):
        p = module_dir / name
        if p.is_file():
            return p
    raise FileNotFoundError(
        f"Missing {stem}.csv or {stem}.csv.gz under {module_dir}"
    )
