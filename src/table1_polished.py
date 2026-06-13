#!/usr/bin/env python3
"""
Generate a polished Table 1 (300-DPI PNG + CSV) from the current cohort.

Run in an interactive HPC session — NOT on the login node:
    interact -n 4 -m 8g -t 00:30:00
    cd /oscar/home/syang195/1595-final
    source .venv/bin/activate
    python src/table1_polished.py
"""

from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parents[1]
COHORT_CSV  = PROJECT_DIR / "cohort.csv"
OUT_DIR     = PROJECT_DIR / "results" / "table1"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Race mapping (39 raw MIMIC values → 6 publication groups) ─────────────────
RACE_MAP: dict[str, str] = {
    "WHITE":                                       "White",
    "WHITE - OTHER EUROPEAN":                      "White",
    "WHITE - RUSSIAN":                             "White",
    "WHITE - EASTERN EUROPEAN":                    "White",
    "WHITE - BRAZILIAN":                           "White",
    "PORTUGUESE":                                  "White",
    "BLACK/AFRICAN AMERICAN":                      "Black / African American",
    "BLACK/CARIBBEAN ISLAND":                      "Black / African American",
    "BLACK/CAPE VERDEAN":                          "Black / African American",
    "BLACK/AFRICAN":                               "Black / African American",
    "MULTIPLE RACE/ETHNICITY":                     "Black / African American",
    "ASIAN":                                       "Asian",
    "ASIAN - CHINESE":                             "Asian",
    "ASIAN - SOUTH EAST ASIAN":                    "Asian",
    "ASIAN - ASIAN INDIAN":                        "Asian",
    "ASIAN - KOREAN":                              "Asian",
    "HISPANIC OR LATINO":                          "Hispanic / Latino",
    "HISPANIC/LATINO - PUERTO RICAN":              "Hispanic / Latino",
    "HISPANIC/LATINO - DOMINICAN":                 "Hispanic / Latino",
    "HISPANIC/LATINO - GUATEMALAN":                "Hispanic / Latino",
    "HISPANIC/LATINO - SALVADORAN":                "Hispanic / Latino",
    "HISPANIC/LATINO - CUBAN":                     "Hispanic / Latino",
    "HISPANIC/LATINO - COLUMBIAN":                 "Hispanic / Latino",
    "HISPANIC/LATINO - HONDURAN":                  "Hispanic / Latino",
    "HISPANIC/LATINO - MEXICAN":                   "Hispanic / Latino",
    "HISPANIC/LATINO - CENTRAL AMERICAN":          "Hispanic / Latino",
    "SOUTH AMERICAN":                              "Hispanic / Latino",
    "UNKNOWN":                                     "Unknown / Not Reported",
    "UNABLE TO OBTAIN":                            "Unknown / Not Reported",
    "PATIENT DECLINED TO ANSWER":                  "Unknown / Not Reported",
    "OTHER":                                       "Other",
    "AMERICAN INDIAN/ALASKA NATIVE":               "Other",
    "NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER":   "Other",
}
RACE_ORDER = [
    "White",
    "Black / African American",
    "Asian",
    "Hispanic / Latino",
    "Unknown / Not Reported",
    "Other",
]


# ── ICU unit mapping (16 raw → 6 publication groups) ──────────────────────────
UNIT_MAP: dict[str, str] = {
    "Medical Intensive Care Unit (MICU)":               "Medical (MICU)",
    "Intensive Care Unit (ICU)":                        "Medical (MICU)",
    "Medicine":                                         "Medical (MICU)",
    "Medical/Surgical Intensive Care Unit (MICU/SICU)": "Med/Surgical (MICU/SICU)",
    "Surgical Intensive Care Unit (SICU)":              "Surgical (SICU/TSICU)",
    "Trauma SICU (TSICU)":                              "Surgical (SICU/TSICU)",
    "Surgery/Vascular/Intermediate":                    "Surgical (SICU/TSICU)",
    "Surgery/Trauma":                                   "Surgical (SICU/TSICU)",
    "Med/Surg":                                         "Surgical (SICU/TSICU)",
    "Cardiac Vascular Intensive Care Unit (CVICU)":     "Cardiovascular (CVICU/CCU)",
    "Coronary Care Unit (CCU)":                         "Cardiovascular (CVICU/CCU)",
    "Medicine/Cardiology Intermediate":                 "Cardiovascular (CVICU/CCU)",
    "Neuro Surgical Intensive Care Unit (Neuro SICU)":  "Neurological ICU",
    "Neuro Intermediate":                               "Neurological ICU",
    "Neuro Stepdown":                                   "Neurological ICU",
    "Neurology":                                        "Neurological ICU",
    "PACU":                                             "Other / PACU",
}
UNIT_ORDER = [
    "Medical (MICU)",
    "Med/Surgical (MICU/SICU)",
    "Surgical (SICU/TSICU)",
    "Cardiovascular (CVICU/CCU)",
    "Neurological ICU",
    "Other / PACU",
]


# ── Formatting helpers ─────────────────────────────────────────────────────────
def _mean_sd(s: pd.Series) -> str:
    return f"{s.mean():.1f} ± {s.std():.1f}"

def _median_iqr(s: pd.Series) -> str:
    return f"{s.median():.1f} [{s.quantile(0.25):.1f}–{s.quantile(0.75):.1f}]"

def _n_pct(n: int, total: int) -> str:
    return f"{n:,} ({n / total * 100:.1f})"

def _fmt_p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"

def _fmt_smd(s: float) -> str:
    return f"{s:.3f}"


# ── SMD helpers ────────────────────────────────────────────────────────────────
def _smd_continuous(a: pd.Series, b: pd.Series) -> float:
    pooled = np.sqrt((a.std() ** 2 + b.std() ** 2) / 2)
    return abs(a.mean() - b.mean()) / pooled if pooled > 0 else 0.0

def _smd_binary(p1: float, p2: float) -> float:
    denom = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / 2)
    return abs(p1 - p2) / denom if denom > 0 else 0.0

def _cramers_v(col: pd.Series, label: pd.Series) -> float:
    ct = pd.crosstab(col, label)
    chi2, _, _, _ = stats.chi2_contingency(ct)
    n = int(ct.values.sum())
    phi2 = chi2 / n
    r, k = ct.shape
    phi2_corr = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    r_corr = r - (r - 1) ** 2 / (n - 1)
    k_corr = k - (k - 1) ** 2 / (n - 1)
    denom = min(k_corr - 1, r_corr - 1)
    return float(np.sqrt(phi2_corr / denom)) if denom > 0 else 0.0

def _p_chi2(col: pd.Series, label: pd.Series) -> float:
    ct = pd.crosstab(col, label)
    _, p, _, _ = stats.chi2_contingency(ct)
    return float(p)

def _p_ttest(a: pd.Series, b: pd.Series) -> float:
    _, p = stats.ttest_ind(a.dropna(), b.dropna(), equal_var=False)
    return float(p)


# ── Row type constants ─────────────────────────────────────────────────────────
TOPLINE = "topline"   # sample-size header row
SECTION = "section"   # bold section header (Demographics / Clinical)
VAR     = "var"       # variable row (has stats + SMD + p-value)
SUBCAT  = "subcat"    # indented sub-category (n % only)


def _row(rt, char, overall="", nd="", d="", smd="", p=""):
    return dict(rt=rt, char=char, overall=overall, nd=nd, d=d, smd=smd, p=p)


# ── Table builder ──────────────────────────────────────────────────────────────
def build_table(df: pd.DataFrame):
    g0 = df["label"] == 0
    g1 = df["label"] == 1
    n  = len(df)
    n0 = int(g0.sum())
    n1 = int(g1.sum())

    rows: list[dict] = []

    # ── Sample sizes ──────────────────────────────────────────────────────────
    rows.append(_row(TOPLINE, "ICU stays, N",
                     overall=f"{n:,}", nd=f"{n0:,}", d=f"{n1:,}"))

    # ── Demographics ──────────────────────────────────────────────────────────
    rows.append(_row(SECTION, "Demographics"))

    # Age
    a0 = df.loc[g0, "anchor_age"].dropna()
    a1 = df.loc[g1, "anchor_age"].dropna()
    rows.append(_row(VAR, "  Age, years — mean ± SD",
                     overall=_mean_sd(df["anchor_age"].dropna()),
                     nd=_mean_sd(a0), d=_mean_sd(a1),
                     smd=_fmt_smd(_smd_continuous(a0, a1)),
                     p=_fmt_p(_p_ttest(a0, a1))))

    # Sex
    m0 = int((df.loc[g0, "gender"] == "M").sum())
    m1 = int((df.loc[g1, "gender"] == "M").sum())
    m  = int((df["gender"] == "M").sum())
    rows.append(_row(VAR, "  Male sex, n (%)",
                     overall=_n_pct(m, n), nd=_n_pct(m0, n0), d=_n_pct(m1, n1),
                     smd=_fmt_smd(_smd_binary(m0 / n0, m1 / n1)),
                     p=_fmt_p(_p_chi2(df["gender"], df["label"]))))

    # Race / Ethnicity
    df["_race"] = df["race"].map(RACE_MAP).fillna("Other")
    rows.append(_row(VAR, "  Race / Ethnicity, n (%)",
                     smd=_fmt_smd(_cramers_v(df["_race"], df["label"])),
                     p=_fmt_p(_p_chi2(df["_race"], df["label"]))))
    for cat in RACE_ORDER:
        if cat not in df["_race"].values:
            continue
        nc  = int((df["_race"] == cat).sum())
        nc0 = int((df.loc[g0, "_race"] == cat).sum())
        nc1 = int((df.loc[g1, "_race"] == cat).sum())
        rows.append(_row(SUBCAT, f"    {cat}",
                         overall=_n_pct(nc, n), nd=_n_pct(nc0, n0), d=_n_pct(nc1, n1)))

    # Insurance
    ins_remap = {"No charge": "Other"}
    df["_ins"] = df["insurance"].fillna("Missing").replace(ins_remap)
    ins_order = [c for c in ["Medicare", "Medicaid", "Private", "Other", "Missing"]
                 if c in df["_ins"].values]
    rows.append(_row(VAR, "  Insurance type, n (%)",
                     smd=_fmt_smd(_cramers_v(df["_ins"], df["label"])),
                     p=_fmt_p(_p_chi2(df["_ins"], df["label"]))))
    for cat in ins_order:
        nc  = int((df["_ins"] == cat).sum())
        nc0 = int((df.loc[g0, "_ins"] == cat).sum())
        nc1 = int((df.loc[g1, "_ins"] == cat).sum())
        rows.append(_row(SUBCAT, f"    {cat}",
                         overall=_n_pct(nc, n), nd=_n_pct(nc0, n0), d=_n_pct(nc1, n1)))

    # Marital status
    ms_map = {"MARRIED": "Married", "SINGLE": "Single",
               "WIDOWED": "Widowed", "DIVORCED": "Divorced"}
    df["_ms"] = df["marital_status"].map(ms_map).fillna("Missing / Unknown")
    ms_order = [c for c in ["Married", "Single", "Widowed", "Divorced", "Missing / Unknown"]
                if c in df["_ms"].values]
    rows.append(_row(VAR, "  Marital status, n (%)",
                     smd=_fmt_smd(_cramers_v(df["_ms"], df["label"])),
                     p=_fmt_p(_p_chi2(df["_ms"], df["label"]))))
    for cat in ms_order:
        nc  = int((df["_ms"] == cat).sum())
        nc0 = int((df.loc[g0, "_ms"] == cat).sum())
        nc1 = int((df.loc[g1, "_ms"] == cat).sum())
        rows.append(_row(SUBCAT, f"    {cat}",
                         overall=_n_pct(nc, n), nd=_n_pct(nc0, n0), d=_n_pct(nc1, n1)))

    # ── Clinical Characteristics ───────────────────────────────────────────────
    rows.append(_row(SECTION, "Clinical Characteristics"))

    # ICU LOS — median [IQR] (right-skewed: report robust summary)
    l0 = df.loc[g0, "los_hours"].dropna()
    l1 = df.loc[g1, "los_hours"].dropna()
    rows.append(_row(VAR, "  ICU LOS, hours — median [IQR]",
                     overall=_median_iqr(df["los_hours"].dropna()),
                     nd=_median_iqr(l0), d=_median_iqr(l1),
                     smd=_fmt_smd(_smd_continuous(l0, l1)),
                     p=_fmt_p(_p_ttest(l0, l1))))

    # First ICU unit
    df["_unit"] = df["first_careunit"].map(UNIT_MAP).fillna("Other / PACU")
    rows.append(_row(VAR, "  First ICU unit, n (%)",
                     smd=_fmt_smd(_cramers_v(df["_unit"], df["label"])),
                     p=_fmt_p(_p_chi2(df["_unit"], df["label"]))))
    for cat in UNIT_ORDER:
        if cat not in df["_unit"].values:
            continue
        nc  = int((df["_unit"] == cat).sum())
        nc0 = int((df.loc[g0, "_unit"] == cat).sum())
        nc1 = int((df.loc[g1, "_unit"] == cat).sum())
        rows.append(_row(SUBCAT, f"    {cat}",
                         overall=_n_pct(nc, n), nd=_n_pct(nc0, n0), d=_n_pct(nc1, n1)))

    return rows, n, n0, n1


# ── CSV export ─────────────────────────────────────────────────────────────────
def save_csv(rows, n, n0, n1, path: Path) -> None:
    records = []
    for r in rows:
        records.append({
            "Characteristic":              r["char"].strip(),
            f"Overall (N={n:,})":          r["overall"],
            f"No Delirium (n={n0:,})":     r["nd"],
            f"Delirium (n={n1:,})":        r["d"],
            "SMD":                         r["smd"],
            "p-value":                     r["p"],
        })
    pd.DataFrame(records).to_csv(path, index=False)
    print(f"  Saved CSV  -> {path}")


# ── Figure constants ──────────────────────────────────────────────────────────
_C = {
    "header_bg":  "#2c3e50",
    "header_fg":  "white",
    "section_bg": "#d6eaf8",
    "section_fg": "#1a252f",
    "topline_bg": "#eaf4fb",
    "even_bg":    "#f4f6f7",
    "odd_bg":     "white",
    "border":     "#2c3e50",
    "text":       "#1a252f",
    "foot":       "#555555",
}

_COL_WIDTHS  = [0.26, 0.135, 0.155, 0.155, 0.092, 0.105]  # fractions, ~sum=0.90
_FONT_BODY   = 10.5
_FONT_HEADER = 11.5
_FOOTNOTE = (
    "Abbreviations: IQR = interquartile range; LOS = length of stay; "
    "MICU = medical ICU; SICU = surgical ICU; "
    "TSICU = trauma surgical ICU; CVICU = cardiac vascular ICU; "
    "CCU = coronary care unit.\n"
    "SMD = standardized mean difference (|SMD| > 0.10 suggests "
    "meaningful imbalance). Cohen’s d for continuous variables; Cramér’s V "
    "for multi-level categorical variables;\n"
    "binary SMD for sex. P-values: Welch’s t-test (continuous), "
    "Pearson χ² (categorical)."
)


# ── PNG export ─────────────────────────────────────────────────────────────────
def save_png(rows, n, n0, n1, path: Path) -> None:
    col_labels = [
        "Characteristic",
        f"Overall\n(N = {n:,})",
        f"No Delirium\n(n = {n0:,})",
        f"Delirium\n(n = {n1:,})",
        "SMD",
        "p-value",
    ]

    cell_text = [
        [r["char"], r["overall"], r["nd"], r["d"], r["smd"], r["p"]]
        for r in rows
    ]

    n_rows = len(rows)
    fig_w  = 10.0
    # Estimate figure height: header + data rows + title + footnote
    fig_h  = max(11.0, 1.0 + n_rows * 0.36 + 0.90)

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

    # ── Title ────────────────────────────────────────────────────────────────
    title_frac = 0.70 / fig_h
    fig.text(
        0.5, 1.0 - 0.12 / fig_h,
        "Table 1.  Baseline Characteristics of ICU Stays Stratified by Delirium Onset",
        ha="center", va="top",
        fontsize=13, fontweight="bold", color=_C["text"],
        fontfamily="DejaVu Sans",
    )

    # ── Axes for table body ──────────────────────────────────────────────────
    foot_frac  = 0.68 / fig_h
    table_frac = 1.0 - title_frac - foot_frac
    ax = fig.add_axes([0.005, foot_frac, 0.990, table_frac])
    ax.axis("off")

    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        colWidths=_COL_WIDTHS,
        loc="center",
        cellLoc="center",
        bbox=[0, 0, 1, 1],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(_FONT_BODY)

    # ── Style column-label row (index 0) ─────────────────────────────────────
    for ci in range(len(col_labels)):
        cell = tbl[0, ci]
        cell.set_facecolor(_C["header_bg"])
        cell.set_text_props(color=_C["header_fg"], fontweight="bold",
                            fontsize=_FONT_HEADER)
        cell.set_edgecolor("none")
        # Bump header row height slightly for two-line labels
        cell.set_height(cell.get_height() * 1.55)

    # ── Style data rows ───────────────────────────────────────────────────────
    data_row_idx = 0
    for ri, r in enumerate(rows):
        tbl_ri = ri + 1
        rt = r["rt"]
        for ci in range(len(col_labels)):
            cell = tbl[tbl_ri, ci]
            cell.set_edgecolor("none")

            if rt == TOPLINE:
                cell.set_facecolor(_C["topline_bg"])
                if ci == 0:
                    cell.set_text_props(fontweight="bold")
            elif rt == SECTION:
                cell.set_facecolor(_C["section_bg"])
                cell.set_text_props(fontweight="bold", color=_C["section_fg"])
            else:
                bg = _C["even_bg"] if data_row_idx % 2 == 0 else _C["odd_bg"]
                cell.set_facecolor(bg)

            # Left-align the Characteristic column; center others
            if ci == 0:
                cell.get_text().set_ha("left")
                cell.get_text().set_x(0.01)
            else:
                cell.get_text().set_ha("center")

        if rt not in (TOPLINE, SECTION):
            data_row_idx += 1

    # ── Horizontal rules (booktabs style) ────────────────────────────────────
    header_frac = 1.0 / (n_rows + 1)
    for y, lw in [(1.0, 1.5), (1.0 - header_frac, 0.8), (0.0, 1.5)]:
        ax.plot([0.0, 1.0], [y, y], transform=ax.transAxes,
                color=_C["border"], linewidth=lw, clip_on=False)

    # ── Footnote ──────────────────────────────────────────────────────────────
    fig.text(
        0.01, 0.005,
        _FOOTNOTE,
        ha="left", va="bottom",
        fontsize=8.5, style="italic", color=_C["foot"],
        fontfamily="DejaVu Sans",
        linespacing=1.35,
    )

    plt.savefig(path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved PNG  -> {path}")


# ── Entry point ────────────────────────────────────────────────────────────────
def main() -> None:
    print(f"Loading {COHORT_CSV} ...")
    df = pd.read_csv(COHORT_CSV)
    df["label"] = df["label"].astype(int)
    n, n_pos = len(df), int(df["label"].sum())
    print(f"  {n:,} stays | {n_pos:,} delirium ({n_pos/n*100:.1f}%)")

    print("Building table ...")
    rows, n, n0, n1 = build_table(df)

    print("Saving outputs ...")
    save_csv(rows, n, n0, n1, OUT_DIR / "table1_polished.csv")
    save_png(rows, n, n0, n1, OUT_DIR / "table1_polished.png")
    print("Done.")


if __name__ == "__main__":
    main()
