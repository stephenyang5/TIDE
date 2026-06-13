"""Build 8-hour patches from features_hourly.csv + cohort.csv for TTCN encoding.

Tensor contract (per sample, before batch collation):

- values (V, P, L_pad) — min–max normalized measurement (0 if padded).
- times (V, P, L_pad) — normalized to [0, 1] as hour_index / T (ICU hour from admission).
- point_mask (V, P, L_pad) — 1 if real observation for that slot.
- patch_mask (V, P) — 1 if the variable has at least one observation in that patch.
- label scalar int (from cohort label).

Optional features_hourly_prelocf.csv (same long schema as hourly, before LOCF in notebook 01)
enables faithful IMTS masks -- otherwise every filled hourly cell is treated as observed.

Performance notes
-----------------
* ICUPatchDataset.__init__ groups features by stay_id into a dict once, so each
  __getitem__ call is O(rows for that stay) rather than O(all rows).
* _obs_sets is built with vectorised pandas ops (no iterrows).
* The patch-building inner loop is replaced by numpy reshape + fancy-index operations.
* Pass features_df (already-loaded DataFrame) to avoid re-reading the CSV when
  constructing train/val/test splits inside train.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.feature_vocab import FEATURE_NAMES, NAME_TO_IDX, NUM_FEATURES

DEFAULT_PATCH_HOURS = 8


def _read_long(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, compression="infer")


def compute_per_feature_minmax(
    features_long: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (min, max) each shape (V,) over FEATURE_NAMES."""
    mins = np.zeros(NUM_FEATURES, dtype=np.float64)
    maxs = np.ones(NUM_FEATURES, dtype=np.float64)
    g = features_long.groupby("feature_name")["value"]
    for name in FEATURE_NAMES:
        if name in g.groups:
            s = g.get_group(name)
            mins[NAME_TO_IDX[name]] = float(s.min())
            maxs[NAME_TO_IDX[name]] = float(s.max())
    maxs = np.where(maxs > mins, maxs, mins + 1.0)
    return mins, maxs


class ICUPatchDataset(Dataset):
    def __init__(
        self,
        cohort_path: Path,
        features_path: Path,
        *,
        patch_hours: int = DEFAULT_PATCH_HOURS,
        max_hours: int | None = 24,
        prelocf_features_path: Path | None = None,
        stay_ids: list[int] | None = None,
        value_mins: np.ndarray | None = None,
        value_maxs: np.ndarray | None = None,
        features_df: pd.DataFrame | None = None,
        cohort_df: pd.DataFrame | None = None,
    ) -> None:
        super().__init__()
        self.patch_hours = patch_hours
        self.max_hours = max_hours
        cohort_path = Path(cohort_path)
        features_path = Path(features_path)

        self.cohort = cohort_df if cohort_df is not None else _read_long(cohort_path)
        self.feats = features_df if features_df is not None else _read_long(features_path)

        if stay_ids is not None:
            sid_set = set(stay_ids)
            self.cohort = self.cohort[self.cohort["stay_id"].isin(sid_set)].reset_index(drop=True)
            if features_df is None:
                # feats was loaded from disk - filter to the requested stay subset
                self.feats = self.feats[self.feats["stay_id"].isin(sid_set)].reset_index(drop=True)
            # If features_df was provided it should already be pre-filtered by caller

        # Restrict to prediction horizon (default: first 24 h, following DeLLiriuM standard)
        if max_hours is not None:
            self.feats = self.feats[self.feats["hour_offset"] < max_hours].reset_index(drop=True)

        # Build stay-indexed lookup for constant access per __getitem__ 
        self._feats_by_stay: dict[int, pd.DataFrame] = {
            int(sid): grp for sid, grp in self.feats.groupby("stay_id")
        }

        # Pre-LOCF observation sets
        pl_path = prelocf_features_path
        if pl_path is None:
            cand = cohort_path.parent / "features_hourly_prelocf.csv"
            pl_path = cand if cand.is_file() else None
        self.prelocf = _read_long(pl_path) if pl_path and Path(pl_path).is_file() else None
        if self.prelocf is not None and max_hours is not None:
            self.prelocf = self.prelocf[
                self.prelocf["hour_offset"] < max_hours
            ].reset_index(drop=True)

        self._obs_sets: dict[int, set[tuple[int, int]]] | None = None
        if self.prelocf is not None:
            self._obs_sets = {}
            for sid, g in self.prelocf.groupby("stay_id"):
                fi = g["feature_name"].map(NAME_TO_IDX)
                valid = fi.notna()
                hs = g.loc[valid, "hour_offset"].to_numpy(dtype=int)
                vs = fi[valid].to_numpy(dtype=int)
                self._obs_sets[int(sid)] = set(zip(hs.tolist(), vs.tolist()))

        # Normalisation stats
        if value_mins is None or value_maxs is None:
            self.value_mins, self.value_maxs = compute_per_feature_minmax(self.feats)
        else:
            self.value_mins = np.asarray(value_mins, dtype=np.float64)
            self.value_maxs = np.asarray(value_maxs, dtype=np.float64)

    def __len__(self) -> int:
        return len(self.cohort)

    def _stay_arrays(self, stay_id: int) -> tuple[np.ndarray, np.ndarray]:
        """Dense (T, V) value and observation arrays — O(rows for this stay)."""
        g = self._feats_by_stay.get(stay_id)
        if g is None or g.empty:
            return np.zeros((0, NUM_FEATURES), np.float32), np.zeros((0, NUM_FEATURES), np.float32)

        T = int(g["hour_offset"].max()) + 1
        vals = np.full((T, NUM_FEATURES), np.nan, dtype=np.float32)

        # Vectorised assignment — map feature names to indices, skip unknowns
        feat_idx = g["feature_name"].map(NAME_TO_IDX)  # pd.Series, NaN for unknowns
        valid = feat_idx.notna()
        hours = g.loc[valid, "hour_offset"].to_numpy(dtype=int)
        fidx = feat_idx[valid].to_numpy(dtype=int)
        raw_vals = g.loc[valid, "value"].to_numpy(dtype=np.float32)
        vals[hours, fidx] = raw_vals

        # Observation mask
        obs = np.zeros((T, NUM_FEATURES), np.float32)
        stay_obs = (self._obs_sets or {}).get(stay_id)
        if stay_obs is not None:
            if stay_obs:
                hs_list, vs_list = zip(*stay_obs)
                hs_a = np.array(hs_list, dtype=int)
                vs_a = np.array(vs_list, dtype=int)
                in_range = hs_a < T
                obs[hs_a[in_range], vs_a[in_range]] = 1.0
        else:
            obs[~np.isnan(vals)] = 1.0

        return vals, obs

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int]:
        row = self.cohort.iloc[idx]
        stay_id = int(row["stay_id"])
        label = int(row["label"])

        los = float(row.get("los_hours", np.nan))
        vals_T, obs_T = self._stay_arrays(stay_id)
        t_los = int(np.ceil(los)) if not np.isnan(los) else 0
        if self.max_hours is not None:
            t_los = min(t_los, self.max_hours)

        if vals_T.shape[0] == 0:
            T = t_los
        else:
            T = max(vals_T.shape[0], t_los)

        if T == 0:
            P    = 0
            Lmax = self.patch_hours
            return {
                "values": torch.zeros(NUM_FEATURES, P, Lmax),
                "times": torch.zeros(NUM_FEATURES, P, Lmax),
                "point_mask": torch.zeros(NUM_FEATURES, P, Lmax),
                "patch_mask": torch.zeros(NUM_FEATURES, P, dtype=torch.float32),
                "stay_patch_mask": torch.zeros(P, dtype=torch.float32),
                "label": label,
                "stay_id": stay_id,
            }

        if vals_T.shape[0] == 0:
            vals_T = np.full((T, NUM_FEATURES), np.nan, dtype=np.float32)
            obs_T = np.zeros((T, NUM_FEATURES), np.float32)
        elif vals_T.shape[0] < T:
            pad = T - vals_T.shape[0]
            vals_T = np.pad(vals_T, ((0, pad), (0, 0)), constant_values=np.nan)
            obs_T = np.pad(obs_T, ((0, pad), (0, 0)), constant_values=0.0)

        P    = T // self.patch_hours
        Lmax = self.patch_hours

        if P == 0:
            return {
                "values": torch.zeros(NUM_FEATURES, 0, Lmax),
                "times": torch.zeros(NUM_FEATURES, 0, Lmax),
                "point_mask": torch.zeros(NUM_FEATURES, 0, Lmax),
                "patch_mask": torch.zeros(NUM_FEATURES, 0, dtype=torch.float32),
                "stay_patch_mask": torch.zeros(0, dtype=torch.float32),
                "label": label,
                "stay_id": stay_id,
            }

        # Vectorised patch construction
        T_trimmed = P * Lmax
        vs3 = vals_T[:T_trimmed].reshape(P, Lmax, NUM_FEATURES)  # (P, L, V)
        ob3 = obs_T[:T_trimmed].reshape(P, Lmax, NUM_FEATURES)

        scale = self.value_maxs - self.value_mins
        scale = np.where(scale > 1e-8, scale, 1.0)

        nv = np.clip((vs3 - self.value_mins) / scale, 0.0, 1.0)
        observed = (ob3 > 0) & ~np.isnan(vs3)
        nv[~observed] = 0.0

        h_idx = np.arange(T_trimmed, dtype=np.float32).reshape(P, Lmax)
        t_norm = (h_idx + 1.0) / max(T, 1) # (P, L)
        t3 = np.broadcast_to(t_norm[:, :, np.newaxis], (P, Lmax, NUM_FEATURES)).copy()

        # (P, L, V) to (V, P, L)
        values = nv.transpose(2, 0, 1).astype(np.float32)
        times_out = t3.transpose(2, 0, 1)
        pm_out = observed.transpose(2, 0, 1).astype(np.float32)

        patch_mask = (pm_out.sum(axis=-1) > 0).astype(np.float32)  # (V, P)
        stay_patch_mask = np.ones(P, dtype=np.float32)

        return {
            "values": torch.from_numpy(values),
            "times": torch.from_numpy(times_out),
            "point_mask": torch.from_numpy(pm_out),
            "patch_mask": torch.from_numpy(patch_mask),
            "stay_patch_mask": torch.from_numpy(stay_patch_mask),
            "label": label,
            "stay_id": stay_id,
        }


def collate_patches(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Pad P and L to batch maxima. Returns batch tensors [B, V, P, L] etc."""
    B = len(batch)
    V = NUM_FEATURES
    max_p = max(int(b["values"].shape[1]) for b in batch)
    max_l = max(int(b["values"].shape[2]) for b in batch)

    values = torch.zeros(B, V, max_p, max_l)
    times = torch.zeros(B, V, max_p, max_l)
    point_mask = torch.zeros(B, V, max_p, max_l)
    patch_mask = torch.zeros(B, V, max_p)
    stay_patch_mask = torch.zeros(B, max_p)
    labels = torch.zeros(B, dtype=torch.long)
    stay_ids: list[int] = []

    for i, b in enumerate(batch):
        p_i, l_i = b["values"].shape[1], b["values"].shape[2]
        if p_i == 0:
            labels[i] = b["label"]
            stay_ids.append(int(b["stay_id"]))
            continue
        values[i, :, :p_i, :l_i] = b["values"]
        times[i, :, :p_i, :l_i] = b["times"]
        point_mask[i, :, :p_i, :l_i] = b["point_mask"]
        patch_mask[i, :, :p_i] = b["patch_mask"]
        stay_patch_mask[i, :p_i] = b["stay_patch_mask"]
        labels[i] = b["label"]
        stay_ids.append(int(b["stay_id"]))

    return {
        "values": values,
        "times": times,
        "point_mask": point_mask,
        "patch_mask": patch_mask,
        "stay_patch_mask": stay_patch_mask,
        "label": labels,
        "stay_id": stay_ids,
    }
