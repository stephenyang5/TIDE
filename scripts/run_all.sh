#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_all.sh — single-entry reproduction of the ICU delirium pipeline.
#
# Stages:
#   0. Environment check
#   1. (manual) Feature extraction notebook  — requires MIMIC-IV access
#   2. Cohort build (optional ICD path; canonical labels come from notebook)
#   3. Train the conservative headline config (no CAM/RASS)
#   4. Evaluate + interpretability
#   5. Tests (synthetic; no MIMIC needed)
#
# Usage:
#   bash scripts/run_all.sh [--full|--conservative|--tests-only]
#
# Defaults to --conservative (the intended honest headline).
# Requires an activated/available .venv and (for stages 1-4) MIMIC-IV on Oscar.
# ---------------------------------------------------------------------------
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"

MODE="${1:---conservative}"

# --- Stage 0: environment ---------------------------------------------------
if [[ -f "$PROJECT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$PROJECT/.venv/bin/activate"
fi
echo "Python : $(python --version 2>&1)"
python -c "import torch; print('torch  :', torch.__version__, '| cuda:', torch.cuda.is_available())"

# --- Stage 5 shortcut: tests only ------------------------------------------
# Prefer pytest; fall back to direct execution (some Oscar login nodes have a
# pytest<->ssl/OpenSSL import mismatch unrelated to project code).
if [[ "$MODE" == "--tests-only" ]]; then
  if python -m pytest tests/ -q 2>/dev/null; then
    exit 0
  fi
  echo "[run_all] pytest unavailable in this env; running test files directly ..."
  python tests/test_patch_encoder.py
  python tests/test_temporal_stack.py
  exit 0
fi

# --- Stage 1: feature extraction (manual / notebook) ------------------------
if [[ ! -f "$PROJECT/features_hourly.csv" || ! -f "$PROJECT/cohort.csv" ]]; then
  cat <<'MSG'
[run_all] features_hourly.csv / cohort.csv not found.
          Run the extraction notebook first (requires MIMIC-IV on Oscar):
            jupyter nbconvert --to notebook --execute 01_cohort_extraction.ipynb
          Then re-run this script.
MSG
  exit 1
fi

# --- Stage 3 + 4 args by mode ----------------------------------------------
if [[ "$MODE" == "--full" ]]; then
  OUT_CK="$PROJECT/checkpoints_full"
  OUT_RES="$PROJECT/results_full"
  EXCL=()
else
  OUT_CK="$PROJECT/checkpoints_no_cam_rass"
  OUT_RES="$PROJECT/results_no_cam_rass"
  EXCL=(--exclude-features cam_icu rass)
fi
mkdir -p "$OUT_CK" "$OUT_RES/interpret" "$PROJECT/logs"

# --- Stage 3: train ---------------------------------------------------------
echo "[run_all] Training ($MODE) ..."
python -m src.train \
  --cohort   "$PROJECT/cohort.csv" \
  --features "$PROJECT/features_hourly.csv" \
  --output-dir "$OUT_CK" \
  --max-hours 24 --epochs 50 --batch-size 32 \
  --hid-dim 32 --n-layer 2 --nhead 4 --tf-layer 2 --node-dim 10 --dropout 0.1 \
  --lr 1e-3 --grad-clip 1.0 --bootstrap-iters 200 \
  --history-csv "$OUT_RES/training_history.csv" \
  --predictions-csv "$OUT_RES/test_predictions.csv" \
  "${EXCL[@]}"

# --- Stage 4: evaluate + interpret -----------------------------------------
echo "[run_all] Interpretability ..."
python -m src.interpret_eval \
  --checkpoint "$OUT_CK/best_model.pt" \
  --cohort "$PROJECT/cohort.csv" \
  --features "$PROJECT/features_hourly.csv" \
  --output-dir "$OUT_RES/interpret" \
  --batch-size 32 --seed 42

echo "[run_all] Done. Outputs in $OUT_RES"
