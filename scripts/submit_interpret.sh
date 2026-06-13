#!/usr/bin/env bash
# Slurm batch script — interpretability suite for the delirium model (Oscar HPC)
#
# Run AFTER training is complete and checkpoints/best_model.pt exists.
#
# Submit:  sbatch scripts/submit_interpret.sh
# Dry-run: sbatch --test-only scripts/submit_interpret.sh
#
#SBATCH --job-name=delirium_interpret
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=logs/interpret_%j.out
#SBATCH --error=logs/interpret_%j.err
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
mkdir -p "$PROJECT/logs"
mkdir -p "$PROJECT/results/interpret"

# ── Verify checkpoint exists ───────────────────────────────────────────────
if [ ! -f "$PROJECT/checkpoints/best_model.pt" ]; then
    echo "ERROR: checkpoint not found at $PROJECT/checkpoints/best_model.pt"
    echo "Run submit_train.sh first."
    exit 1
fi

# ── Run interpretability suite ─────────────────────────────────────────────
# Remove --skip-ig to include Integrated Gradients (adds ~30 min on CPU,
# ~5 min on GPU for 50 steps over the test set).
python -m src.interpret_eval \
    --checkpoint  "$PROJECT/checkpoints/best_model.pt" \
    --cohort      "$PROJECT/cohort.csv" \
    --features    "$PROJECT/features_hourly.csv" \
    --output-dir  "$PROJECT/results/interpret" \
    --batch-size  32 \
    --ig-steps    50 \
    --val-frac    0.10 \
    --test-frac   0.10 \
    --seed        42

echo ""
echo "Interpretability outputs saved to: $PROJECT/results/interpret/"
echo ""
echo "Files produced:"
ls -lh "$PROJECT/results/interpret/"
