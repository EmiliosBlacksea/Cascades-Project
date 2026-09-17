
# Cascades-Project#
Early Prediction of Information Cascades

**Predicting cascade growth from the first few retweets.**

This project studies how early activity in a Weibo information cascade can help predict its eventual size. It extracts temporal, structural, and user-count features from the first **k retweet events**, compares logistic regression with a small neural network, and evaluates the contribution of individual features through ablation experiments.

The implementation covers raw-data parsing, incremental tree reconstruction, feature extraction, PyTorch training, evaluation, and experiment reporting.

## Prediction task

For each observation size `k`, the pipeline:

1. Keeps cascades with at least `k` valid retweet events.
2. Computes `f(k)`, the **upper median** of their final event counts.
3. Extracts features using only the first `k` events in chronological order.
4. Assigns label `1` when the final event count is at least `f(k)`, and `0` otherwise.

This is **binary classification of cascade size relative to a data-derived threshold**, rather than direct prediction of the final retweet count. Ties at the threshold can make the classes imbalanced.

The code defines final size as the number of parsed events after removing the synthetic root event. It does not use the input file's declared `retweet_number` field as the target count.

## Features

The default configuration extracts **15 features** per cascade snapshot.

| Group | Features | What they capture |
| --- | --- | --- |
| Temporal — 7 | `time_k`, `mean_inter_first_half`, `mean_inter_second_half`, `interarrival_trend`, `burstiness_cv`, `mean_interarrival`, `std_interarrival` | Time to the kth event, spacing between events, changes in activity, and burstiness |
| Structural — 6 | `max_depth_k`, `avg_depth_k`, `p90_depth_k`, `root_outdeg_k`, `num_leaves_k`, `max_outdeg_k` | Depth and branching of the reconstructed propagation tree |
| User proxies — 2 | `unique_leaf_users_k`, `duplicate_leaf_users_k` | Distinct retweeting users and repeated users among the observed events |

Tree reconstruction is incremental: each newly observed user is attached to the nearest already-observed ancestor in its recorded path, falling back to the root. Repeated users do not add new tree nodes. User proxies use event user IDs; no profile or follower-network data is required.

## Models and evaluation

| Model | Architecture |
| --- | --- |
| Logistic regression | A single linear layer producing one logit |
| MLP | Input → 64-unit hidden layer → ReLU → dropout (0.1) → one output logit |

Both models use PyTorch, Adam, and binary cross-entropy with logits. Features are standardized using statistics fitted on the training split only.

Training uses an approximately **70% / 15% / 15%** stratified train/validation/test split. The best checkpoint is selected by validation ROC-AUC, falling back to accuracy when AUC is undefined. Test metrics are **accuracy, F1, and ROC-AUC**; binary predictions use a probability threshold of `0.5`.

CLI defaults are 100 maximum epochs, early-stopping patience of 10, batch size 256, learning rate `1e-3`, and weight decay `1e-4`.

## Setup

Use a Python 3.10+ environment. Run the following commands from a terminal:

```bash
git clone https://github.com/EmiliosBlacksea/Cascades-Project.git
cd Cascades-Project
python -m venv .venv
```

Activate the environment:

```bash
# Linux / macOS
source .venv/bin/activate
```

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install the dependencies used by the Python pipeline:

```bash
python -m pip install numpy pandas pyarrow tqdm scikit-learn torch matplotlib openpyxl
```

These packages are inferred from the source imports; the repository does not currently include a pinned dependency file. MATLAB is optional and is needed only for `src/plotss.m`.

## Dataset

The repository includes a [dataset format description](data/rawdata/README%20(2).md), which describes 119,313 Weibo messages published on June 1, 2016, with retweet counts measured over 24 hours. **The raw dataset itself is not included**, and the repository does not provide a download link.

Once obtained separately, place the input file at:

```text
data/rawdata/weibo_dataset.txt
```

Each line contains five tab-separated fields:

```text
<message_id> <root_user_id> <publish_time> <retweet_number> <retweets>
```

The spaces between the five fields above represent actual **tab characters**. Within the final field, events are separated by spaces and have the form:

```text
user1/user2/.../userN:retweet_time
```

The feature extractor interprets `retweet_time` as an **offset in seconds**, not as an absolute Unix timestamp. Events are sorted by that value, missing roots are prepended to paths, and synthetic root records such as `root_user_id:0` are removed.

## Run the pipeline

Run all commands below from the repository root.

### 1. Preprocess cascade snapshots

```bash
python -m src.preprocess --input data/rawdata/weibo_dataset.txt --out_dir data/processed --ks 5 10 25 50 100
```

Preprocessing reads the data twice: first to compute the thresholds, then to build labeled feature tables. It writes one Parquet file per observation size, such as `data/processed/features_k5.parquet`.

For a smaller trial, add `--max_lines 10000`. This also computes thresholds from that subset, so it changes the resulting task. Each selected `k` must retain enough examples of both classes for stratified splitting.

### 2. Train a model

```bash
python -m src.train --data data/processed/features_k5.parquet --model logreg --out_dir reports/logreg_k5
python -m src.train --data data/processed/features_k5.parquet --model mlp --out_dir reports/mlp_k5
```

Training runs on CPU by default. Use `--device cuda` if a compatible CUDA-enabled PyTorch installation and GPU are available. Additional options include `--epochs`, `--patience`, `--batch_size`, `--lr`, `--weight_decay`, and `--seed`.

### 3. Compare models across observation sizes

```bash
python -m src.run_experiments --processed_dir data/processed --ks 5 10 25 50 100 --models logreg mlp
```

This runner writes to `reports/` and skips missing Parquet files. Alternatively, use `--k_min 5 --k_max 100` to request every integer `k` in that range; the corresponding feature files must already exist.

### 4. Plot baseline results

```bash
python -m src.plot_results --metrics_dir reports/metrics --out_dir reports/plots
```

This generates `summary.csv`, `accuracy_vs_k.png`, and `auc_vs_k.png`. The plotter reads baseline metric filenames and excludes feature-drop variants.

## Ablation experiments

### Leave one feature out

Train both models with all features, then repeat training while dropping each feature individually:

```bash
python -m src.run_experiments --processed_dir data/processed --ks 5 10 25 50 100 --loo
```

To run a specific ablation and retain its checkpoint separately:

```bash
python -m src.train --data data/processed/features_k5.parquet --model mlp --drop_feature time_k --out_dir reports/mlp_k5_drop_time
```

Repeat `--drop_feature` to remove multiple columns in a single run.

### Compare feature groups

Generate strictly temporal-only or structural-only datasets by disabling both other groups:

```bash
# Temporal only
python -m src.preprocess --input data/rawdata/weibo_dataset.txt --out_dir data/processed_temporal --ks 5 10 25 50 100 --no_structural --no_user_proxy

# Structural only
python -m src.preprocess --input data/rawdata/weibo_dataset.txt --out_dir data/processed_structural --ks 5 10 25 50 100 --no_temporal --no_user_proxy
```

Then train on the generated files with a separate output directory for each configuration.

### Longer experiment batches

```bash
python -m src.laptop_stress_test --processed_dir data/processed --k_min 1 --k_max 200 --models logreg mlp --device cpu --skip_existing
```

This sequential runner performs baselines, individual feature ablations, and a joint ablation of all temporal features. It logs experiment status, continues after failed runs, and can skip runs whose metric files already exist. It does not generate missing feature files or verify that existing metrics match the current code and settings.

## Outputs and analysis

| Output | Contents |
| --- | --- |
| `data/processed/features_k<K>.parquet` | Features, labels, message IDs, event counts, and thresholds |
| `<out_dir>/metrics/features_k<K>_<model>.json` | Baseline test metrics, split sizes, feature count, and best epoch |
| `<out_dir>/metrics/features_k<K>_<model>_drop_<feature>.json` | Feature-ablation metrics |
| `<out_dir>/checkpoints/features_k<K>_<model>.pt` | Model weights, ordered feature names, and standardization statistics |
| `reports/plots/` | Baseline comparison plots and CSV summary |

**Checkpoint filenames do not include the dropped features.** Runs sharing the same output directory, `k`, and model overwrite that checkpoint, even though baseline and ablation metrics have separate filenames. Use separate `--out_dir` values with `src.train` when preserving individual models matters.

The repository also includes five `ablation_summary*.xlsx` workbooks. To export newly generated JSON metrics, edit `results_dir`, `out_xlsx`, and the desired range in `src/ablation_to_excel.py`, then run:

```bash
python -m src.ablation_to_excel
```

The exporter currently matches `logreg` filenames; change its model suffix to export MLP results. `src/plotss.m` provides optional MATLAB plots of the workbook results and feature-removal performance differences; configure its input workbook before running it.

## Repository guide

| File | Purpose |
| --- | --- |
| `src/parse.py` | Parse raw cascade records and normalize retweet events |
| `src/features.py` | Extract temporal, tree-structure, and user-proxy features |
| `src/preprocess.py` | Compute thresholds and write labeled Parquet tables |
| `src/datasets.py` | Standardization and PyTorch dataset wrappers |
| `src/models.py` | Logistic regression and MLP definitions |
| `src/train.py` | Train, select checkpoints, and record test metrics |
| `src/eval.py` | Accuracy, F1, and ROC-AUC evaluation |
| `src/run_experiments.py` | Baseline and leave-one-feature-out experiments |
| `src/laptop_stress_test.py` | Sequential batch experiments with logging and skip support |
| `src/plot_results.py` | Baseline plots and CSV summaries |
| `src/ablation_to_excel.py` | Export ablation metrics to Excel |
| `src/plotss.m` | MATLAB analysis of exported workbooks |
| `collab.ipynb` | Colab launcher using a user-specific Google Drive path |

## Experimental considerations

- **Threshold construction:** `f(k)` is computed before splitting and uses all eligible cascades, including those later assigned to validation and test sets. For a strictly held-out evaluation, determine thresholds from training data and apply them to the other splits.
- **Randomness:** `--seed` controls the data splits, but the current training code does not seed PyTorch initialization, dropout, or batch shuffling. Repeated runs can therefore produce different metrics.
- **Comparison across k:** Both the eligible cascade population and the classification threshold change with `k`. Performance curves do not evaluate an identical population and target at every observation size.
- **Memory use:** Preprocessing reads the input line by line but retains event counts and feature rows in memory. Large datasets or many requested observation sizes can require substantial RAM.
- **Scope:** This repository provides an experimental training and evaluation pipeline. It does not currently include a standalone inference CLI, a pinned environment, or an automated test suite.
