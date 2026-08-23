# triternatum_ML

A modular machine learning pipeline for combined genomic sequence and morphological image classification. Processes FASTA alignments and specimen images to train, tune, and evaluate multiple classifiers with publication-ready outputs.

## Features

- One-hot encoding of genomic sequences (FASTA input)
- ResNet-50 morphological feature extraction from images
- SelectKBest feature selection applied independently to genomic and morphological features
- 10 classifiers: ExtraTrees, RandomForest, XGBoost, GradientBoosting, SVM, LogisticRegression, KNN, NaiveBayes, LDA, QDA
- Bayesian hyperparameter tuning via Optuna (stratified 5-fold CV, weighted F1 objective)
- Unsupervised analysis: PCA, t-SNE, UMAP projections; GMM and DBSCAN clustering
- Outputs: confusion matrices, embedding plots, clustering overview, metrics CSV/JSON, saved models

---

## Installation

Requires Python 3.12+.

**With uv (recommended):**
```bash
uv sync
```

**With pip:**
```bash
pip install -e .
```

Core dependencies: `numpy`, `pandas`, `scikit-learn`, `tensorflow`, `xgboost`, `optuna`, `joblib`, `biopython`, `umap-learn`.

---

## Input Data

Three inputs are required:

### 1. FASTA file
A multiple sequence alignment in FASTA format. Each record's description must follow the format `id, Clade name`:
```
>specimen_001, Clade 1
ATCGATCG...
>specimen_002, Clade 2
GCTAGCTA...
```

### 2. Mapping CSV
A two-column CSV (no header) mapping FASTA names to image filenames:
```
specimen_001,specimen_001_corrected
specimen_002,specimen_002
```
Column order: `new_names, old_names`. Image files will be resolved as `{img_prefix}/{old_name}.jpg`.

### 3. Image directory
A folder containing `.jpg` images named `{old_name}.jpg` matching the second column of the mapping CSV.

### 4. Label map (optional)
A JSON file mapping clade names to integer class labels:
```json
{
  "Clade 1": 1,
  "Clade 2": 2,
  "Clade 3": 3
}
```
If omitted, the default label map in `src/gen_class/read_data.py` is used. Samples not found in the label map are assigned label `9` (treated as an "other" class).

---

## Usage

### Command-line

```bash
python main3.py \
  --fasta data/alignment.fasta \
  --mapping-csv data/name_mapping.csv \
  --img-prefix data/images/ \
  --results-dir publication_results/my_run
```

All CLI arguments:

| Argument | Default | Description |
|---|---|---|
| `--fasta` | required | Path to FASTA alignment file |
| `--mapping-csv` | required | Path to name-mapping CSV (new_name, old_name) |
| `--img-prefix` | required | Directory containing specimen images |
| `--label-map-json` | None | Optional JSON file overriding the default label map |
| `--results-dir` | `publication_results` | Output directory |
| `--k-genomic` | 1000 | Top-K genomic features to retain (SelectKBest) |
| `--k-morph` | 500 | Top-K morphological features to retain (SelectKBest) |
| `--n-trials` | 30 | Optuna trials per model |
| `--parallel-trials` | 4 | Parallel Optuna workers |
| `--model-jobs` | auto | `n_jobs` for scikit-learn/XGBoost models |
| `--batch-size` | 32 | Batch size for ResNet-50 feature extraction |

### Python API

```python
from gen_class import PublicationReadyClassifier

classifier = PublicationReadyClassifier(
    results_dir="my_results",
    k_genomic=1000,
    k_morph=500,
    n_trials=30,
    parallel_trials=4,
)

outputs = classifier.run_complete_pipeline(
    fasta_path="data/alignment.fasta",
    mapping_csv_path="data/name_mapping.csv",
    img_prefix="data/images/",
    batch_size=32,
)

# Access results
for name, result in outputs.baseline_results.items():
    print(name, result.metrics)

for name, result in outputs.tuned_results.items():
    print(name, result.metrics, result.best_params)

for name, res in outputs.unsupervised_results.items():
    print(name, f"ARI={res.ari:.3f}  silhouette={res.silhouette:.3f}")
```

Step-by-step execution:
```python
classifier = PublicationReadyClassifier(results_dir="my_results")

# Step 1: load FASTA + images
classifier.load_data(
    fasta_path="data/alignment.fasta",
    mapping_csv_path="data/name_mapping.csv",
    img_prefix="data/images/",
)

# Step 2: extract and select features
classifier.prepare_features(batch_size=32)

# Step 3: train, tune, cluster, save
outputs = classifier.train_and_tune()
```

---

Evaluation metrics recorded per model: accuracy, weighted F1, macro F1, precision, recall, MCC, and 5-fold cross-validation mean/std.

---

## Package Structure

```
src/gen_class/
├── __init__.py          # exports PublicationReadyClassifier, PipelineOutputs
├── pipeline.py          # main orchestrator
├── config.py            # constants, model configs, hyperparameter grids
├── read_data.py         # FASTA + mapping CSV → DataFrame
├── data_loader.py       # sequence encoding, image preprocessing, LoadedData
├── features.py          # ResNet-50 extraction, SelectKBest, FeatureSet
├── modeling.py          # baseline training, Optuna tuning, ModelResult
├── evaluation.py        # metric computation
├── unsupervised.py      # PCA/t-SNE/UMAP projections, GMM, DBSCAN
├── results.py
```

---

## Configuration

Edit `src/gen_class/config.py` to adjust:

- **Model parameters**: `MODELS_CONFIG` — default hyperparameters for each classifier
- **Tuning search space**: `PARAM_GRIDS` — ranges explored by Optuna
- **Feature selection**: `FEATURE_SELECTION_K_RANGE`, `MIN_FEATURE_VARIANCE`
- **Clustering**: `DBSCAN_EPS_VALUES`, `GMM_N_COMPONENTS_RANGE`
- **Train/test split**: `TEST_SIZE` (default 0.2), `CV_FOLDS` (default 5)
- **Plot quality**: `FIGURE_DPI` (default 300)

To override which samples are dropped before analysis, edit the `drop_names` set in `src/gen_class/read_data.py`.

---

## Troubleshooting

**Sequence length mismatch** — the pipeline requires a pre-aligned FASTA; all sequences must be the same length.

**Images not found** — verify the mapping CSV `old_names` column matches the image filenames (without `.jpg`), and that `--img-prefix` points to the correct directory.

**Memory issues during feature extraction** — reduce `--batch-size` (e.g. `--batch-size 8`) or decrease `--k-genomic` / `--k-morph`.

**Slow tuning** — reduce `--n-trials` (e.g. `--n-trials 10`) or `--parallel-trials 1` on machines with limited cores.
