"""CLI: build the demographic ICU cohort (LOS / age / first-ICU / death / ICD exclusions).

This produces the unlabeled cohort. The delirium label is assigned
separately from CAM-ICU + RASS chart events 

"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.cohort import build_cohort, load_exclusion_hadm_ids
from src.mimic_paths import (
    admissions_path,
    diagnoses_icd_path,
    icustays_path,
    patients_path,
    mimic_root,
)


def _default_output_dir() -> Path:
    root = Path(__file__).resolve().parents[1]
    return root / "results"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="MIMIC-IV demographic ICU cohort (unlabeled).")
    p.add_argument(
        "--mimic-root", type=Path, default=None,
        help="Override MIMIC root (else MIMIC_ROOT env or Oscar default).",
    )
    p.add_argument("--min-los-hours", type=float, default=24.0)
    p.add_argument("--min-age", type=int, default=18)
    p.add_argument(
        "--skip-exclusions", action="store_true",
        help="Do not scan diagnoses_icd for dementia/TBI/psych exclusions (faster).",
    )
    p.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output CSV path (.csv or .csv.gz). "
             "Default: results/cohort_base_los_ge{hours}h.csv.gz",
    )
    args = p.parse_args(argv)

    if args.mimic_root is not None:
        import os

        os.environ["MIMIC_ROOT"] = str(args.mimic_root)

    root = mimic_root()
    icu_path = icustays_path()
    adm_path = admissions_path()
    pat_path = patients_path()

    for path, label in (
        (icu_path, "icustays"),
        (adm_path, "admissions"),
        (pat_path, "patients"),
    ):
        if not path.is_file():
            print(f"Missing {label} file: {path}", file=sys.stderr)
            print(f"MIMIC root resolved to: {root}", file=sys.stderr)
            return 1

    print(f"Loading icustays from {icu_path}")
    icustays = pd.read_csv(icu_path, compression="infer")
    print(f"Loading admissions from {adm_path}")
    admissions = pd.read_csv(adm_path)
    print(f"Loading patients from {pat_path}")
    patients = pd.read_csv(pat_path)

    exclusion_ids: set[int] | None = None
    if not args.skip_exclusions:
        dpath = diagnoses_icd_path()
        if not dpath.is_file():
            print(f"Missing diagnoses_icd: {dpath}", file=sys.stderr)
            return 1
        print("Scanning diagnoses_icd for dementia/TBI/psych exclusions …")
        exclusion_ids = load_exclusion_hadm_ids()

    cohort = build_cohort(
        icustays,
        admissions,
        patients,
        exclusion_hadm_ids=exclusion_ids,
        min_los_hours=args.min_los_hours,
        min_age=args.min_age,
    )

    out = args.output
    if out is None:
        out_dir = _default_output_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        los = args.min_los_hours
        h = int(los) if los == int(los) else los
        out = out_dir / f"cohort_base_los_ge{h}h.csv.gz"

    out.parent.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(out, index=False, compression="infer")
    print(f"Wrote {len(cohort):,} rows to {out}")
    print("NOTE: this cohort is UNLABELED. Assign the delirium label from "
          "CAM-ICU + RASS via src.labeling")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
