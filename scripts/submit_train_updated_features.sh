#!/usr/bin/env bash
# Slurm batch script — delirium training with cam_icu + rass excluded
#
# Motivation: cam_icu values are always 0 in the feature window (any CAM+
# in hours 0-24 causes cohort exclusion), so the model learns from
# assessment *frequency* via point_mask rather than score values — a form
# of ascertainment leakage.  RASS is co-documented with CAM-ICU and carries
# the same proxy signal.  This run removes both to obtain an honest AUROC.
#
# Submit:  sbatch scripts/submit_train_updated_features.sh
# Dry-run: sbatch --test-only scripts/submit_train_updated_features.sh
#
#SBATCH --job-name=delirium_feature_updated
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_updated_features_%j.out
#SBATCH --error=logs/train_no_cam_rass_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=syang195@brown.edu

set -euo pipefail

PROJECT=/oscar/home/syang195/1595-final
cd "$PROJECT"

# ── Activate virtual environment ───────────────────────────────────────────
source "$PROJECT/.venv/bin/activate"
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"

# ── GPU diagnostics ────────────────────────────────────────────────────────
echo "SLURM_JOB_ID:   $SLURM_JOB_ID"
echo "SLURM_NODELIST: $SLURM_NODELIST"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "
import torch
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Device:', torch.cuda.get_device_name(0))
"

# ── Create output directories ──────────────────────────────────────────────
mkdir -p "$PROJECT/checkpoints_updated_features"
mkdir -p "$PROJECT/results_updated_features/interpret"
mkdir -p "$PROJECT/logs"

# ── Train ──────────────────────────────────────────────────────────────────
python -m src.train \
    --cohort       "$PROJECT/cohort.csv" \
    --features     "$PROJECT/features_hourly.csv" \
    --output-dir   "$PROJECT/checkpoints_updated_features" \
    --max-hours    24 \
    --epochs       50 \
    --batch-size   32 \
    --hid-dim      32 \
    --n-layer      2 \
    --nhead        4 \
    --tf-layer     2 \
    --node-dim     10 \
    --dropout      0.1 \
    --lr           1e-3 \
    --grad-clip    1.0 \
    --lr-factor    0.5 \
    --lr-patience  5 \
    --lr-min       1e-5 \
    --patience     10 \
    --min-delta    1e-4 \
    --bootstrap-iters 200 \
    --history-csv  "$PROJECT/results_updated_features/training_history.csv" \
    --predictions-csv "$PROJECT/results_updated_features/test_predictions.csv"

echo "Training complete."
echo "Checkpoint  : $PROJECT/checkpoints_updated_features/best_model.pt"
echo "History     : $PROJECT/results_updated_features/training_history.csv"
echo "Predictions : $PROJECT/results_updated_features/test_predictions.csv"

# ── Generate visualisation plots ──────────────────────────────────────────
python -c "
from src.viz import make_all_plots
make_all_plots(
    'results_updated_features/test_predictions.csv',
    'results_updated_features/training_history.csv',
    output_dir='results_updated_features',
    show=False,
)
"
echo "Plots saved to $PROJECT/results_updated_features/"

# ── Interpretability suite ────────────────────────────────────────────────
echo ""
echo "Running interpretability suite ..."
python -m src.interpret_eval \
    --checkpoint  "$PROJECT/checkpoints_updated_features/best_model.pt" \
    --cohort      "$PROJECT/cohort.csv" \
    --features    "$PROJECT/features_hourly.csv" \
    --output-dir  "$PROJECT/results_updated_features/interpret" \
    --batch-size  32 \
    --ig-steps    50 \
    --val-frac    0.10 \
    --test-frac   0.10 \
    --seed        42
echo "Interpretability outputs saved to $PROJECT/results_updated_features/interpret/"
