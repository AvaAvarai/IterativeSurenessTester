# Bidirectional Active Processing (BAP) Implementation

Reimplementation of the BAP algorithm from [Bidirectional Active Processing.md](./docs/algorithms/bidirectional%20active%20processing/Bidirectional%20Active%20Processing.md) as specified in our Electronics paper available both [online](https://www.mdpi.com/2079-9292/15/3/580) and [locally](docs/bidirectional%20active%20processing/Quantifying%20AI%20Model%20Trust%20as%20a%20Model%20Sureness%20Measure%20by%20Bidirectional%20Active%20Processing%20and%20Visual%20Knowledge%20Discovery.pdf). Single-file implementations in Python and Julia with TOML configuration.

## Requirements

### Python
- Python 3.11  
- `numpy`, `pandas`, `scikit-learn`  
- `tomli` (for Python &lt; 3.11)  

Use the project venv when present (dependencies are installed there):

```bash
source .venv311/bin/activate   # macOS / Linux
python bap.py -c config_iris.toml
```

Or run without activating:

```bash
.venv311/bin/python bap.py -c config_iris.toml
```

To create or refresh the venv:

```bash
python3.11 -m venv .venv311
.venv311/bin/python -m pip install -r requirements.txt
```

Configs point at CSVs under [computing/machine learning datasets](../../machine%20learning%20datasets/) (not copies inside other app repos).

### Julia
- Julia 1.8+  
- Packages: CSV, DataFrames, ScikitLearn, TOML  
  (ScikitLearn requires Python with scikit-learn)

```bash
julia -e 'using Pkg; Pkg.activate("."); Pkg.instantiate()'
```
(Run from this `BAP rebuild` directory.)

## Quick Start

### Single CSV (train/test split)
Example: Fisher Iris dataset
```bash
# Python (after: source .venv311/bin/activate)
python bap.py -c config_iris.toml

# Or with flags
python bap.py --train "../../machine learning datasets/default/fisher_iris.csv" --testing split --split 0.8,0.2 --classifier dt -t 0.95 -n 10 -m 5
```

### Separate train and test CSVs
Example: MNIST
```bash
# Python
python bap.py -c config_mnist.toml

# Or with flags
python bap.py --train "../../machine learning datasets/default/mnist_train.csv" --test "../../machine learning datasets/default/mnist_test.csv" --testing fixed --classifier knn --k 5 -n 5 -m 100
```

### Julia
```bash
julia bap.jl -c config_iris.toml
julia bap.jl -c config_mnist.toml
julia bap.jl --train "../../machine learning datasets/default/fisher_iris.csv" -c config_iris.toml  # override train path
```

## Configuration

All options can be set via **TOML config** (preferred) or **CLI flags**. TOML overrides defaults; flags override TOML.

### TOML structure

| Parameter | TOML | Definition |
|-----------|------|------------|
| `train` | `train = "path.csv"` | Training CSV (or single dataset to split) |
| `test` | `test = "path.csv"` | Test CSV (for testing.fixed) |
| `testing` | `[testing.fixed]` or `[testing.split]` | How to obtain test set |
| `testing.fixed` | `[testing.fixed]` + `test = "..."` | Use separate test file |
| `testing.split` | `[testing.split]` + `split = [0.8, 0.2]` | Split train ratio : test ratio |
| `classifier` | `classifier = "dt"` | `dt`, `knn`, or `svm` |
| `parameters` | `[parameters]` + `k = 5` | Classifier hyperparameters (e.g. `k` for KNN) |
| `distance` | `distance = "euclidean"` | Distance metric for KNN |
| `goal.t` | `[goal]` + `t = 0.95` | Accuracy threshold (0–1) |
| `direction` | `[direction.forward]` or `[direction.backward]` | Forward (additive) or backward (subtractive) |
| `splits` | `splits = 1` | Number of train/test splits |
| `n` | `n = 10` | Iterations per split |
| `m` | `m = 5` | Cases added/removed per iteration |
| `sampling` | `[sampling.stratified]` or `[sampling.random]` | Sampling method |
| `seed` | `seed = 42` | PRNG seed |
| `output_dir` | `output_dir = "results"` | Output directory |

### CLI flags (Python)

| Flag | Description |
|------|-------------|
| `-c`, `--config` | TOML config file |
| `--train` | Training CSV |
| `--test` | Test CSV (for fixed) |
| `--testing` | `fixed` \| `split` \| `cv` |
| `--split` | Train,test ratio, e.g. `0.8,0.2` |
| `--classifier` | `dt` \| `knn` \| `svm` |
| `--k` | K for KNN (default 3) |
| `--distance` | Metric for KNN |
| `-t`, `--threshold` | Accuracy threshold |
| `--direction` | `forward` \| `backward` |
| `--splits` | Number of splits |
| `-n`, `--iterations` | Iterations per split |
| `-m` | Cases per iteration |
| `--sampling` | `random` \| `stratified` |
| `--seed` | Random seed |
| `-o`, `--output-dir` | Output directory |

## Data format

CSVs must have a **class column** whose header matches `class`, `label`, or `target` (**case-insensitive**).  
All other columns are features (column order may follow your benchmark file, e.g. `fisher_iris.csv`).

- **fisher_iris.csv**: `class` column  
- **mnist_train.csv**, **mnist_test.csv**: `label` column  

## Classifiers

| Code | Classifier |
|------|------------|
| `dt` | Decision Tree |
| `knn` | K-Nearest Neighbors |
| `svm` | Support Vector Machine (RBF) |
| `hb_vis` | Hyperblock (VisCanvas-style) |
| `hb_dv` | Hyperblock (DV-style, interval-based) |

## Output

Results are written to `{output_dir}/bap_{timestamp}/` (default `results/bap_YYYYMMDD_HHMMSS/`).

Exported **case** and **hyperblock** CSVs use the same tabular shape as the shared datasets (e.g. `computing/machine learning datasets/default/fisher_iris.csv`):

- One header **`class`** (lowercase), case label for data rows.  
- One column per **attribute** (same names as the training CSV). **No** separate `*_min` / `*_max` columns and **no** extra ID column.  
- **`split_N/converged_exp_{id}_seed{seed}.csv`** – converged training cases: `class` holds the dataset label (e.g. `Setosa`).  
- **`split_N/converged_exp_{id}_seed{seed}_hyperblocks.csv`** – when using `hb_vis` or `hb_dv`: **two rows per hyperblock**. Each row has the same attribute columns; values are the box **minimum** (`…__bottom`) and **maximum** (`…__top`) corners. The `class` cell encodes label, HB id, and edge, e.g. `Setosa__HB0__bottom` / `Setosa__HB0__top`. Any `__` in the dataset label is replaced by `_` so the suffix pattern stays parseable.  

Other output:

- `config.txt` – Settings used  
- `statistics.csv` – Aggregate statistics (mean/min/max cases, convergence rate, etc.)  

Every converged result CSV has a matching `_hyperblocks.csv` in the same directory when using hyperblock classifiers.

## Algorithm (summary)

1. Set PRNG seed  
2. For each split: load data (fixed test or split)  
3. For each iteration: start with empty set (forward) or full set (backward)  
4. While accuracy &lt; threshold and cases remain: add/remove `m` cases via sampling  
5. Record converged subsets and compute statistics  
