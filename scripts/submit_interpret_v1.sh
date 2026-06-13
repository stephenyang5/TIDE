#!/usr/bin/env bash
# Phase 3 interpretability on v1 checkpoints (permutation + graph heterogeneity + IG).
#
# Submit:  sbatch scripts/submit_interpret_v1.sh
#
#SBATCH --job-name=delirium_interpret_v1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=logs/interpret_v1_%j.out
#SBATCH --error=logs/interpret_v1_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=syang195@brown.edu

set -euo pipefail

PROJECT=/oscar/home/syang195/1595-final
cd "$PROJECT"
export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"

source "$PROJECT/.venv/bin/activate"
mkdir -p "$PROJECT/logs"

run_interp() {
  local cfg="$1"
  local ckpt="$2"
  local out="$3"
  echo "=================== interpret v1: $cfg ==================="
  python -m src.interpret_eval \
    --checkpoint "$ckpt" \
    --cohort "$PROJECT/cohort.csv" \
    --features "$PROJECT/features_hourly.csv" \
    --output-dir "$out" \
    --batch-size 32 --ig-steps 50 --perm-repeats 10 --seed 42
}

run_interp conservative \
  "$PROJECT/checkpoints_v1_conservative/best_model.pt" \
  "$PROJECT/results_v1_conservative/interpret"

run_interp no_cam_rass \
  "$PROJECT/checkpoints_v1_no_cam_rass/best_model.pt" \
  "$PROJECT/results_v1_no_cam_rass/interpret"

run_interp full \
  "$PROJECT/checkpoints_v1_full/best_model.pt" \
  "$PROJECT/results_v1_full/interpret"

echo "All v1 interpretability done."
