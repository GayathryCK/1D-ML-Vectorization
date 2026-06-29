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

ModelNet40: 9,843 train / 2,468 test, 40 categories, 1,024 points per shape

Ground-truth entropy labels (H0, H1, H2) must be pre-computed and provided as CSV files with columns: `sample_id`, `class_name`, `class_idx`, `H0`, `H1`, `H2`.

---

## Repository Structure

```
RipsNet_ModelNet40_Entropy/
├── train_ripsnet_entropy.py       — training script
├── predict_ripsnet_entropy.py     — inference script
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

### 1. Train RipsNet — ModelNet40

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

### 2. Predict with Trained Model — ModelNet40

```bash
python predict_ripsnet_entropy.py \
    --weights_path ripsnet_entropy_out/ripsnet_entropy_best.weights.h5 \
    --test_csv     ground_truths/mn40_test_entropy.csv \
    --data_root    ModelNet40 \
    --out_csv      ripsnet_entropy_out/ripsnet_test_predictions.csv \
    --num_points   1024
```
