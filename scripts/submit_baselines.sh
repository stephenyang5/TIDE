#!/usr/bin/env bash
# Slurm batch script — classical baselines (LogReg + HistGBT), all 3 configs.
#
# Baselines are CPU-only (scikit-learn), so this requests a CPU partition and
# multiple cores (HistGradientBoosting parallelizes across them). No GPU.
#
# Submit:  sbatch scripts/submit_baselines.sh
# Dry-run: sbatch --test-only scripts/submit_baselines.sh
#
#SBATCH --job-name=delirium_baselines
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/baselines_%j.out
#SBATCH --error=logs/baselines_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=syang195@brown.edu

set -euo pipefail

PROJECT=/oscar/home/syang195/1595-final
cd "$PROJECT"
# Make `src` importable regardless of the cwd Slurm starts in.
export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"
# Let scikit-learn use all allocated cores.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

source "$PROJECT/.venv/bin/activate"
echo "Python : $(python --version 2>&1)"
echo "Job    : ${SLURM_JOB_ID:-<none>} on ${SLURM_NODELIST:-<local>}"
mkdir -p "$PROJECT/logs" "$PROJECT/results/baselines"

for cfg in full no_cam_rass conservative; do
  echo "=================== baseline: $cfg ==================="
  python -m src.baselines \
    --config "$cfg" \
    --cohort   "$PROJECT/cohort.csv" \
    --features "$PROJECT/features_hourly_prelocf.csv" \
    --max-hours 24 \
    --n-boot 200 \
    --output-dir "$PROJECT/results/baselines"
done

echo "All baselines done → $PROJECT/results/baselines/"
