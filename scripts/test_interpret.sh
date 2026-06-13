#!/usr/bin/env bash
# Quick smoke test for the interpretability module — runs on a CPU node.
# Tests all imports, model changes, and one mini forward pass per method.
# Does NOT require MIMIC data or a checkpoint.
#
# Submit: sbatch scripts/test_interpret.sh
#
#SBATCH --job-name=delirium_test_interpret
#SBATCH --partition=batch
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:15:00
#SBATCH --output=logs/test_interpret_%j.out
#SBATCH --error=logs/test_interpret_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=syang195@brown.edu

set -euo pipefail

PROJECT=/oscar/home/syang195/1595-final
cd "$PROJECT"

source "$PROJECT/.venv/bin/activate"
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
mkdir -p "$PROJECT/logs"

python - <<'PYEOF'
import sys, tempfile, torch, numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from src.models.delirium_backbone import DeliriumClassifier
from src.models.temporal_adaptive_stack import TemporalAdaptiveGNNStack
from src.data.feature_vocab import NUM_FEATURES
from src.interpret.feature_ablation import baseline_auroc, run_feature_ablation, run_patch_ablation, plot_feature_importance
from src.interpret.graph_viz import extract_adjacency, plot_adjacency
from src.interpret.attention import aggregate_attention, plot_attention
from src.interpret.embed_viz import extract_embeddings, plot_tsne
from src.interpret.integrated_gradients import aggregate_ig, plot_ig_heatmap

print("All imports OK")

# ── Verify model changes ───────────────────────────────────────────────────
stack = TemporalAdaptiveGNNStack(d_model=16, n_layer=1, nhead=2, tf_layer=1)
assert hasattr(stack, '_graph_cache') and stack._graph_cache is None
print("TemporalAdaptiveGNNStack._graph_cache: OK")

model = DeliriumClassifier(hid_dim=16, n_layer=1, nhead=2, tf_layer=1, node_dim=4)
assert hasattr(model, 'forward_explain')
print("DeliriumClassifier.forward_explain: OK")

torch.manual_seed(42)
model.eval()
device = torch.device('cpu')
V, P = NUM_FEATURES, 3

def make_batch(B=8):
    return {
        'values':          torch.randn(B, V, P, 8).clamp(0, 1),
        'times':           torch.rand(B, V, P, 8),
        'point_mask':      torch.ones(B, V, P, 8),
        'patch_mask':      torch.ones(B, V, P),
        'stay_patch_mask': torch.ones(B, P),
        'label':           torch.randint(0, 2, (B,)),
        'stay_id':         list(range(B)),
    }

batches = [make_batch() for _ in range(3)]
loader = DataLoader(batches, batch_size=None, collate_fn=lambda x: x)
outdir = Path(tempfile.mkdtemp())

# forward_explain shapes
with torch.no_grad():
    out = model.forward_explain(make_batch(4))
assert out['logit'].shape == (4, 1)
assert out['patch_embeddings'].shape == (4, V, P, 16)
assert out['embedding'].shape == (4, 16)
print("forward_explain shapes: OK")

# graph cache
model.backbone.stack._graph_cache = []
with torch.no_grad():
    model(make_batch(4))
assert len(model.backbone.stack._graph_cache) == 1
assert model.backbone.stack._graph_cache[0].shape == (4, P, V, V)
model.backbone.stack._graph_cache = None
print("_graph_cache capture: OK")

# feature ablation (2 features only)
ref = baseline_auroc(model, loader, device)
from src.interpret.feature_ablation import ablate_feature, ablate_patch
a0 = ablate_feature(model, loader, 0, device)
a1 = ablate_patch(model, loader, 0, device)
print(f"Feature ablation: OK  (ref={ref:.3f}, feat0={a0:.3f}, patch0={a1:.3f})")

# graph extraction + plot
adp = extract_adjacency(model, loader, device)
assert adp['pos'].shape == (1, V, V)
plot_adjacency(adp, outdir)
print(f"Graph extraction: OK  shape={adp['pos'].shape}")

# attention
attn = aggregate_attention(model, loader, device)
if attn is not None:
    assert attn.shape[1:] == (2, P, P)  # (n_calls, nhead, P, P)
    plot_attention(attn, outdir)
    print(f"Attention extraction: OK  shape={attn.shape}")
else:
    print("Attention: no weights captured (skipped)")

# embeddings + t-SNE (small N, fast)
embs, lbls, probs = extract_embeddings(model, loader, device)
assert embs.shape == (24, 16)  # 3 batches × 8
plot_tsne(embs, lbls, probs, outdir, perplexity=5)
print(f"Embeddings: OK  shape={embs.shape}")

# IG (2 steps only for speed)
attrs, ig_lbls = aggregate_ig(model, loader, n_steps=2, device=device)
assert attrs.shape == (24, V, P, 8)
plot_ig_heatmap(attrs, ig_lbls, outdir)
print(f"IG attribution: OK  shape={attrs.shape}")

files = sorted(outdir.iterdir())
print(f"\nAll {len(files)} output files generated:")
for f in files:
    print(f"  {f.name}  ({f.stat().st_size:,} B)")

print("\nAll tests passed.")
PYEOF

echo ""
echo "Test complete."
