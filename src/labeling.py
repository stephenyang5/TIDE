"""DeLLiriuM-compatible delirium labeling from CAM-ICU + RASS chart events.

This module is the single source of truth for the delirium label.

Label definition (Contreras et al., 2025, "DeLLiriuM"):
    Delirium onset = a CAM-ICU positive assessment together with a
    RASS >= −3 (patient assessable, not comatose) occurring within the same
    12-hour assessment interval, where the interval starts after the
    first prediction_window_hours (24 h) of the ICU stay.

Cohort criteria derived from the same chart signals:

  Prevalent delirium: any CAM+ within the first 24 h - exclude
  
  Coma in first 24 h: persistently comatose (max RASS <= −4 over all
    first-24 h readings) -  exclude (patient never assessable for delirium).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# itemids 
CAM_ICU_ITEMID = 228332
RASS_ITEMID = 228096

# label / selection thresholds
DEFAULT_PREDICTION_WINDOW_HOURS = 24.0
DEFAULT_ASSESSMENT_INTERVAL_HOURS = 12.0
RASS_ASSESSABLE_MIN = -3.0 # RASS >= −3 -> assessable for CAM-ICU
RASS_COMA_MAX = -4.0 # RASS <= −4 -> comatose / unarousable

CAM_POSITIVE_STRINGS = {"positive", "pos", "yes", "y", "1", "true"}


# scaling helpers 

def cam_is_positive(value: pd.Series, valuenum: pd.Series | None = None) -> pd.Series:
    """Boolean Series: True where a CAM-ICU row indicates a positive screen.

    Accepts free-text value (e.g. "Positive") and/or numeric valuenum
    (1.0 == positive in MIMIC charting for this item).
    """
    txt = value.astype(str).str.strip().str.lower().isin(CAM_POSITIVE_STRINGS)
    if valuenum is not None:
        num = pd.to_numeric(valuenum, errors="coerce") == 1
        return (txt | num.fillna(False)).astype(bool)
    return txt.astype(bool)


def _interval_index(hour_offset: pd.Series, window: float, interval: float) -> pd.Series:
    """0-based 12 h assessment-interval index for times after the window."""
    return np.floor((hour_offset - window) / interval).astype("Int64")


# cohort criteria

def prevalent_delirium_stays(
    cam_events: pd.DataFrame,
    *,
    prediction_window_hours: float = DEFAULT_PREDICTION_WINDOW_HOURS,
) -> set[int]:
    """stay_ids with any CAM+ at or before the prediction window - prevalent."""
    if cam_events.empty:
        return set()
    m = cam_events["is_positive"] & (cam_events["hour_offset"] <= prediction_window_hours)
    return set(cam_events.loc[m, "stay_id"].astype(int))


def coma_first24_stays(
    rass_events: pd.DataFrame,
    *,
    prediction_window_hours: float = DEFAULT_PREDICTION_WINDOW_HOURS,
    coma_rass_max: float = RASS_COMA_MAX,
    policy: str = "persistent",
) -> set[int]:
    """stay_ids comatose in the first window.

    policy="persistent" - default: the patient's maximum RASS in the first
    window is <= coma_rass_max (i.e. never arousable above coma) — the
    defensible DeLLiriuM-style criterion.
    policy="any": exclude if any first-window RASS <= coma_rass_max.
    """
    if rass_events.empty:
        return set()
    first = rass_events[rass_events["hour_offset"] <= prediction_window_hours]
    if first.empty:
        return set()
    if policy == "any":
        m = first["rass_val"] <= coma_rass_max
        return set(first.loc[m, "stay_id"].astype(int))
    if policy == "persistent":
        max_rass = first.groupby("stay_id")["rass_val"].max()
        return set(max_rass.index[max_rass <= coma_rass_max].astype(int))
    raise ValueError(f"unknown coma policy: {policy!r}")


# label construction 
def assessed_after_window(
    cam_events: pd.DataFrame,
    rass_events: pd.DataFrame | None = None,
    *,
    prediction_window_hours: float = DEFAULT_PREDICTION_WINDOW_HOURS,
    assessment_interval_hours: float = DEFAULT_ASSESSMENT_INTERVAL_HOURS,
    min_intervals: int = 1,
) -> set[int]:
    """stay_ids that were actually screened after the prediction window.

    A stay counts as assessed when it has at least min_intervals distinct
    12 h post-window intervals containing a CAM-ICU assessment (
    """
    frames = []
    if cam_events is not None and not cam_events.empty:
        frames.append(cam_events[["stay_id", "hour_offset"]])
    if rass_events is not None and not rass_events.empty:
        frames.append(rass_events[["stay_id", "hour_offset"]])
    if not frames:
        return set()
    ev = pd.concat(frames, ignore_index=True)
    ev = ev[ev["hour_offset"] > prediction_window_hours].copy()
    if ev.empty:
        return set()
    ev["interval"] = _interval_index(
        ev["hour_offset"], prediction_window_hours, assessment_interval_hours
    )
    n_intervals = ev.groupby("stay_id")["interval"].nunique()
    return set(n_intervals.index[n_intervals >= min_intervals].astype(int))


def label_delirium(
    cam_events: pd.DataFrame,
    rass_events: pd.DataFrame,
    stay_ids: list[int] | set[int],
    *,
    prediction_window_hours: float = DEFAULT_PREDICTION_WINDOW_HOURS,
    assessment_interval_hours: float = DEFAULT_ASSESSMENT_INTERVAL_HOURS,
    rass_assessable_min: float = RASS_ASSESSABLE_MIN,
) -> pd.DataFrame:
    """Assign the delirium label to each stay.

    Parameters
    ----------
    cam_events : DataFrame with columns stay_id, hour_offset, is_positive.
    rass_events : DataFrame with columns stay_id, hour_offset, rass_val.
    stay_ids : the cohort stays to label (stays with no positive interval → 0).

    Returns
    -------
    DataFrame indexed by stay_id with columns:
      label (int 0/1) and first_delirium_interval_start (hours from
      admission, NaN for negatives).

    A stay is positive iff there exists a 12 h assessment interval 
    containing both a CAM+ event and a RASS >= rass_assessable_min reading.
    """
    sid_index = pd.Index(sorted(int(s) for s in stay_ids), name="stay_id")
    out = pd.DataFrame(index=sid_index)
    out["label"] = 0
    out["first_delirium_interval_start"] = np.nan

    if cam_events.empty or rass_events.empty:
        return out.reset_index()

    cam = cam_events[
        cam_events["is_positive"]
        & (cam_events["hour_offset"] > prediction_window_hours)
    ].copy()
    rass = rass_events[
        (rass_events["rass_val"] >= rass_assessable_min)
        & (rass_events["hour_offset"] > prediction_window_hours)
    ].copy()
    if cam.empty or rass.empty:
        return out.reset_index()

    cam["interval"] = _interval_index(
        cam["hour_offset"], prediction_window_hours, assessment_interval_hours
    )
    rass["interval"] = _interval_index(
        rass["hour_offset"], prediction_window_hours, assessment_interval_hours
    )

    # An interval is confirmed when a CAM+ and an assessable RASS share it.
    cam_intervals = cam[["stay_id", "interval"]].drop_duplicates()
    rass_intervals = rass[["stay_id", "interval"]].drop_duplicates()
    confirmed = cam_intervals.merge(rass_intervals, on=["stay_id", "interval"], how="inner")
    if confirmed.empty:
        return out.reset_index()

    # First confirmed interval per stay - its start time (hours from admission).
    first_interval = confirmed.groupby("stay_id")["interval"].min()
    onset = prediction_window_hours + first_interval * assessment_interval_hours

    pos_ids = [s for s in first_interval.index if s in out.index]
    out.loc[pos_ids, "label"] = 1
    out.loc[pos_ids, "first_delirium_interval_start"] = onset.loc[pos_ids].to_numpy()
    return out.reset_index()
