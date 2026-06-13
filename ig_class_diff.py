import numpy as np
from src.data.feature_vocab import FEATURE_NAMES

attrs  = np.load("results_updated_features/interpret/ig_attrs.npy")
labels = np.load("results_updated_features/interpret/ig_labels.npy")

flat = np.abs(attrs).reshape(len(attrs), len(FEATURE_NAMES), -1)
mean_pos = flat[labels == 1].mean(axis=0)
mean_neg = flat[labels == 0].mean(axis=0)
diff = (mean_pos - mean_neg).mean(axis=1)

print("--- More attributive in DELIRIUM patients (positive) ---")
for name, d in sorted(zip(FEATURE_NAMES, diff), key=lambda x: -x[1]):
    if d > 0:
        print(f"  {d:+.6f}  {name}")

print("\n--- More attributive in NO-DELIRIUM patients (negative) ---")
for name, d in sorted(zip(FEATURE_NAMES, diff), key=lambda x: x[1]):
    if d < 0:
        print(f"  {d:+.6f}  {name}")
