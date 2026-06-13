#!/usr/bin/env bash
# Slurm batch script — T-PatchGNN v1 retrain, all 3 canonical configs.
#
# Trains on the fixed v1 cohort (26,345 stays) with the same split / HPs as
# baselines. Outputs land in checkpoints_v1_* / results_v1_* (distinct from
# stale v0 checkpoints).
#
# Submit:  sbatch scripts/submit_train_v1.sh
# Dry-run: sbatch --test-only scripts/submit_train_v1.sh
#
#SBATCH --job-name=delirium_train_v1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_v1_%j.out
#SBATCH --error=logs/train_v1_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=syang195@brown.edu

set -euo pipefail

PROJECT=/oscar/home/syang195/1595-final
cd "$PROJECT"
export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"

source "$PROJECT/.venv/bin/activate"
echo "Python : $(python --version 2>&1)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "Job    : ${SLURM_JOB_ID:-<none>} on ${SLURM_NODELIST:-<local>}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "
import torch
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Device:', torch.cuda.get_device_name(0))
"

mkdir -p "$PROJECT/logs"

COMMON=(
  --cohort       "$PROJECT/cohort.csv"
  --features     "$PROJECT/features_hourly.csv"
  --max-hours    24
  --epochs       50
  --batch-size   32
  --hid-dim      32
  --n-layer      2
  --nhead        4
  --tf-layer     2
  --node-dim     10
  --dropout      0.1
  --lr           1e-3
  --grad-clip    1.0
  --lr-factor    0.5
  --lr-patience  5
  --lr-min       1e-5
  --patience     10
  --min-delta    1e-4
  --bootstrap-iters 200
)

train_config() {
  local cfg="$1"
  local ckpt_dir="$2"
  local res_dir="$3"
  shift 3
  local -a excl=("$@")

  echo "=================== train v1: $cfg ==================="
  mkdir -p "$ckpt_dir" "$res_dir/interpret"

  python -m src.train \
    "${COMMON[@]}" \
    --output-dir "$ckpt_dir" \
    --history-csv "$res_dir/training_history.csv" \
    --predictions-csv "$res_dir/test_predictions.csv" \
    "${excl[@]}"

  python -c "
from src.viz import make_all_plots
make_all_plots(
    '$res_dir/test_predictions.csv',
    '$res_dir/training_history.csv',
    output_dir='$res_dir',
    show=False,
)
"

  python -m src.interpret_eval \
    --checkpoint "$ckpt_dir/best_model.pt" \
    --cohort "$PROJECT/cohort.csv" \
    --features "$PROJECT/features_hourly.csv" \
    --output-dir "$res_dir/interpret" \
    --batch-size 32 \
    --ig-steps 50 \
    --val-frac 0.10 \
    --test-frac 0.10 \
    --seed 42

  echo "Done $cfg → $res_dir"
}

train_config full \
  "$PROJECT/checkpoints_v1_full" \
  "$PROJECT/results_v1_full"

train_config no_cam_rass \
  "$PROJECT/checkpoints_v1_no_cam_rass" \
  "$PROJECT/results_v1_no_cam_rass" \
  --exclude-features cam_icu rass

train_config conservative \
  "$PROJECT/checkpoints_v1_conservative" \
  "$PROJECT/results_v1_conservative" \
  --exclude-features cam_icu rass gcs_eye gcs_verbal gcs_motor

echo "All v1 deep-model configs done."
