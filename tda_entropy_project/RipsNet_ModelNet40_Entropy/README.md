# RipsNet: DeepSet Entropy Regressor for 3D Point Clouds

---

## Prerequisites

- Python 3.13.0
- CPU or GPU (tested on AWS EC2, CPU-only)

---

## Requirements

```
tensorflow==2.21.0
numpy>=2.0.0
pandas>=2.0.0
tqdm>=4.60.0
```

---

## Installation

```bash
pip install tensorflow numpy pandas tqdm
```

---

## Dataset

ModelNet40: 9,843 train / 2,468 test, 40 categories

Ground-truth entropy labels (H0, H1, H2) must be pre-computed and provided as CSV files with columns: `sample_id`, `class_name`, `class_idx`, `H0`, `H1`, `H2`.

---

## Repository Structure

```
RipsNet_ModelNet40_Entropy/
├── train_ripsnet_entropy.py       — training script (used for entropy and amplitude)
├── predict_ripsnet_entropy.py     — inference script
├── run_training.sh                — train RipsNet on ModelNet40 entropy
├── run_all_amplitudes.sh          — train RipsNet on all 6 ModelNet40 amplitude metrics
├── run_sonn_entropy.sh            — train RipsNet on ScanObjectNN entropy
├── run_sonn_amplitudes.sh         — train RipsNet on all 7 ScanObjectNN amplitude metrics
├── run_all_50ep.sh                — 50-epoch variants of all the above
├── ripsnet-main/
│   └── utils.py                  — DenseRagged + PermopRagged layer definitions
└── results/
    ├── ripsnet_entropy_best.weights.h5   — trained weights (best epoch)
    ├── best_test_predictions.csv         — predicted vs. ground truth (2,468 test samples)
    ├── best_test_metrics.json            — MAE / RMSE / R² at best epoch
    ├── training_history.csv              — per-epoch metrics across 100 epochs
    ├── ripsnet_train_config.json         — hyperparameter config
    └── ripsnet_results_summary.txt       — results summary
```

---

## Run Instructions

The same training script handles both entropy and amplitude targets — the target type is determined by the ground truth CSV passed in.

### 1. Train RipsNet — ModelNet40 Entropy

```bash
bash run_training.sh
```

Or manually:

```bash
python train_ripsnet_entropy.py \
    --data_root  ModelNet40 \
    --train_csv  ground_truths/mn40_train_entropy.csv \
    --test_csv   ground_truths/mn40_test_entropy.csv \
    --out_dir    ripsnet_entropy_out \
    --epochs     100 \
    --batch_size 32 \
    --lr         5e-4 \
    --num_points 1024 \
    --seed       42
```

### 2. Train RipsNet — ModelNet40 Amplitude (all metrics)

```bash
bash run_all_amplitudes.sh
```

Trains sequentially over: `betti`, `heat`, `landscape`, `persistence_image`, `silhouette`, `wasserstein`. Output written to `amplitudes/amplitude_<metric>/ripsnet_out/`.

### 3. Train RipsNet — ScanObjectNN Entropy

```bash
bash run_sonn_entropy.sh
```

### 4. Train RipsNet — ScanObjectNN Amplitude (all metrics)

```bash
bash run_sonn_amplitudes.sh
```

Trains sequentially over: `bottleneck`, `betti`, `heat`, `landscape`, `persistence_image`, `silhouette`, `wasserstein`.

### 5. Predict with Trained Model

```bash
python predict_ripsnet_entropy.py \
    --weights_path ripsnet_entropy_out/ripsnet_entropy_best.weights.h5 \
    --test_csv     ground_truths/mn40_test_entropy.csv \
    --data_root    ModelNet40 \
    --out_csv      ripsnet_entropy_out/ripsnet_test_predictions.csv \
    --num_points   1024
```
