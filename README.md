# ICU Delirium Onset Prediction with T-PatchGNN

This project predicts ICU delirium onset from the first 24 hours of physiological monitoring data by adapting T-PatchGNN — an irregular multivariate time-series architecture from ICML 2024 — for binary classification on MIMIC-IV. The approach is motivated by two key readings: Zhang et al. (2024) introduce T-PatchGNN, which segments sparse, irregularly-sampled time series into fixed-size patches and processes them through a combination of time-aware convolution (TTCN), per-patch adaptive graph construction, and intra-series Transformer attention, making it well-suited to the heterogeneous rhythms of ICU charting; and the DeLLiriuM paper (2025) establishes the clinical benchmark (AUROC ~82.5 via an LLM, vs. ~78.1 for structured-EHR baselines) and mandates the precise cohort criteria and label definition used here — delirium onset is a CAM-ICU-positive assessment with RASS ≥ −3 occurring after the first 24 hours, excluding patients with prevalent delirium, early death, dementia, or TBI. The full pipeline covers cohort extraction from MIMIC-IV, 57-feature engineering across vitals, labs, and sedative/vasopressor drugs, 8-hour patch construction with faithful observation masking, class-weighted training, and evaluation with bootstrap confidence intervals.

## Readings

| File | Description |
|------|-------------|
| `readings/zhang24bw.pdf` | The T-PatchGNN paper (Zhang et al., ICML 2024) introducing the patch-based irregular multivariate time-series architecture that this project adapts for delirium classification. |
| `readings/nihpp-rs7216692v1.pdf` | The DeLLiriuM paper (2025) defining the clinical benchmark, delirium label (CAM-ICU + RASS ≥ −3), DeLLiriuM-compliant cohort exclusion criteria, and SOTA AUROC targets. |
| `readings/README.md` | Detailed implementation notes covering the architecture, cohort criteria, feature vocabulary, training protocol, and paper-to-code mappings. |

## Source: Data Pipeline

| File | Description |
|------|-------------|
| `src/mimic_paths.py` | Centralizes MIMIC-IV path resolution, reading from `/oscar/data/shared/ursa/mimic-iv` or a `MIMIC_ROOT` environment variable, with helpers for each module (ICU, HOSP, ED, NOTE). |
| `src/cohort.py` | Builds the DeLLiriuM-compliant patient cohort by merging `icustays`, `admissions`, and `patients`, then enforcing criteria: first ICU stay only, age ≥ 18, LOS ≥ 24 h, no death within 48 h, and optional ICD-based delirium labeling. |
| `src/build_cohort.py` | CLI wrapper for `cohort.py` that scans `diagnoses_icd` for delirium-related ICD codes and writes the cohort to a compressed CSV. |
| `src/data/feature_vocab.py` | Defines the canonical, immutably ordered list of 57 features (16 chart, 24 labs, 17 drugs) and the `NAME_TO_IDX` mapping used for consistent tensor indexing throughout the project. |
| `src/data/patch_dataset.py` | Implements `ICUPatchDataset`, which converts long-format hourly features into `(V, P, L)` patch tensors with three-level masking (point, patch, stay), computes train-split-only min-max normalization, and provides `collate_patches()` for dynamic-length batching. |

## Source: Model Architecture

| File | Description |
|------|-------------|
| `src/models/time_embedding.py` | Implements `LearnableTimeEmbedding`, mapping continuous observation timestamps in [0, 1] to a fixed-dimensional representation via a learnable linear term plus sinusoidal periodic terms. |
| `src/models/ttcn.py` | Implements `TTCN` (Time-aware Temporal Convolutional Network), the meta-filter module that generates adaptive convolution kernels conditioned on patch content and applies them with softmax-masked aggregation over variable-length patches. |
| `src/models/patch_encoder.py` | Implements `PatchTTCNEncoder`, combining `LearnableTimeEmbedding` and `TTCN` to encode raw `(B, V, P, L)` input tensors into `(B, V, P, D)` patch embeddings, with observation masks concatenated to the output. |
| `src/models/gcn.py` | Provides graph convolution building blocks — `NConv` (node-wise einsum convolution), `Conv1x1` (channel-mixing MLP), and `GCN` — supporting multi-order diffusion for the adaptive inter-series graph layer. |
| `src/models/positional_encoding.py` | Provides standard sinusoidal `PositionalEncoding` for the Transformer operating over the patch sequence dimension. |
| `src/models/temporal_adaptive_stack.py` | Implements `TemporalAdaptiveGNNStack`, stacking `n_layer` blocks that alternate between a `TransformerEncoder` for intra-series (temporal) dependencies and an adaptive GCN with dynamically gated per-patch adjacency matrices for inter-series (cross-variable) dependencies. |
| `src/models/delirium_backbone.py` | Assembles the full model: `DeliriumTPatchBackbone` chains `PatchTTCNEncoder` → `TemporalAdaptiveGNNStack`, and `DeliriumClassifier` adds masked mean-pooling over patches and variables, dropout, and a linear head trained with class-weighted `BCEWithLogitsLoss`. |

## Training

| File | Description |
|------|-------------|
| `src/train.py` | End-to-end training script performing stratified 80/10/10 splits, train-set normalization, class-weighted BCE loss, Adam optimization with `ReduceLROnPlateau`, early stopping on validation AUROC, and final test evaluation with 200-iteration bootstrap confidence intervals. |

## Tests

| File | Description |
|------|-------------|
| `tests/test_patch_encoder.py` | Unit tests for `PatchTTCNEncoder` covering output shapes, gradient flow, dataset collation, and the effect of pre-LOCF data on the faithful `point_mask`. |
| `tests/test_temporal_stack.py` | Unit tests for `TemporalAdaptiveGNNStack`, `DeliriumTPatchBackbone`, and `DeliriumClassifier`, verifying shapes, gradients, and masked pooling behavior on synthetic data. |

## Notebooks

| File | Description |
|------|-------------|
| `01_cohort_extraction.ipynb` | Runs the full data pipeline: loads MIMIC-IV tables, applies DeLLiriuM cohort criteria, extracts all 57 features from chart/lab/drug tables, applies LOCF, and writes `features_hourly.csv` and `features_hourly_prelocf.csv`. |
| `02_eda.ipynb` | Exploratory data analysis of patient demographics, feature distributions, missingness patterns, and class balance in the extracted cohort. |
| `03_table1.ipynb` | Generates Table 1 baseline characteristics stratified by delirium label for clinical reporting. |
| `04_train_eval.ipynb` | Orchestrates model training (if not using the CLI) and visualizes results including ROC curves, precision-recall curves, and confusion matrices. |

## Data Files

| File | Description |
|------|-------------|
| `cohort.csv` | Patient cohort after DeLLiriuM-compliant filtering, with one row per ICU stay and columns for `stay_id`, `label`, demographics, and `los_hours`. |
| `features_hourly.csv` | Long-format ICU observations (columns: `stay_id`, `hour_offset`, `feature_name`, `value`) after LOCF imputation; this is the primary model input. |
| `features_hourly_prelocf.csv` | Same schema as `features_hourly.csv` but before LOCF, loaded by `ICUPatchDataset` to construct the faithful `point_mask` distinguishing real observations from filled values. |
| `raw_features_extracted.csv` | Intermediate raw feature extraction output before any imputation or processing, retained for debugging and auditing. |
| `results/table1/table1_baseline_by_label.csv` | Summary statistics table of baseline patient characteristics stratified by delirium label, generated by `03_table1.ipynb`. |

## Configuration

| File | Description |
|------|-------------|
| `CLAUDE.md` | Claude Code project guidance including architecture notes, data pipeline details, paper-to-code mappings, and implementation constraints for AI-assisted development. |

## How to Run the Pipeline

### Prerequisites

- Access to the Oscar HPC cluster
- MIMIC-IV data at `/oscar/data/shared/ursa/mimic-iv` (or set `MIMIC_ROOT` env var)
- A Python virtual environment with PyTorch, scikit-learn, pandas, and Jupyter installed

### 1. Start an interactive GPU session

```bash
interact -q gpu -g 1 -m 20g -n 4 -t 12:00:00
```

### 2. Build the patient cohort (run once)

```bash
python -m src.build_cohort --min-los-hours 24 -o cohort.csv
```

Applies exclusion criteria and writes `cohort.csv` with binary delirium labels.

### 3. Extract features

Open and run `01_cohort_extraction.ipynb` to extract 57 features from MIMIC-IV chart, lab, and drug tables and produce:
- `features_hourly.csv` — post-LOCF observations used for training
- `features_hourly_prelocf.csv` — pre-LOCF observations used for faithful masking

### 4. Explore the data (optional)

Run `02_eda.ipynb` for feature distributions and missingness patterns, and `03_table1.ipynb` to generate baseline characteristics.

### 5. Train the model

```bash
python -m src.train \
  --cohort cohort.csv \
  --features features_hourly.csv \
  --hid-dim 32 \
  --n-layer 2 \
  --epochs 50 \
  --batch-size 32
```

Outputs:
- `checkpoints/best_model.pt` — best checkpoint by validation AUROC
- `results/training_history.csv` — per-epoch metrics
- `results/test_predictions.csv` — test-set predictions with AUROC/AUPRC and 95% bootstrap CI

### 6. Evaluate and visualize

Open `04_train_eval.ipynb` to generate ROC curves, precision-recall curves, and confusion matrices from the saved predictions.

### 7. Run tests

```bash
python -m pytest tests/
```

