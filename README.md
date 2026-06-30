# pcd2Tvec: Efficient Approximation of Topological Descriptors from Point Clouds via Learned Representations

---

## Prerequisites

- Python 3.10.0
- CUDA 12.1 (tested on NVIDIA RTX 4060 8GB)

---

## Requirements

```
numpy==1.26.4
scipy==1.15.3
scikit-learn==1.3.2
pandas==2.3.3
matplotlib==3.10.7
tqdm==4.67.1
joblib==1.5.2
torch==2.5.1+cu121
torchvision==0.20.1+cu121
torchaudio==2.5.1+cu121
giotto-tda==0.6.2
giotto-ph==0.2.4
pyflagser==0.4.7
open3d==0.19.0
```

---

## Installation

```bash
pip install -r requirements.txt
```

For PyTorch (CUDA 12.1):
```bash
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121
```

---

## Datasets

ModelNet40: 9,843 train / 2,468 test, 40 categories, 2,048 points per shape

ScanObjectNN: 11,416 train / 2,882 test, 15 categories, 2,048 points per shape

---

## Repository Structure

```
.
├── tda_amplitude_project/
│   ├── .gitkeep
│   ├── evaluate_vector_structure.py
│   ├── generate_dataset_vectors.py
│   ├── plot_pptnet_vs_ripsnet_testresults.py
│   ├── pptnet_vector_regressor.py
│   ├── train_pptnet_vectors.py
│   └── vector_utils.py
│
├── tda_entropy_project/
│   ├── README.md
│   ├── compute_entropy.py
│   ├── generate_vr_pds.py
│   ├── plot_pptnet_vs_ripsnet_testentropyresults.py
│   ├── pptnet_entropy_regressor.py
│   └── train_pptnet_entropy.py
│
├── README.md
├── evaluate_vector_structure.py
└── plot.py
```

---

## Run Instructions

Supported datasets: `modelnet40`, `sonn`

Supported amplitude metrics: `bottleneck`, `wasserstein`, `betti`, `landscape`, `silhouette`, `heat`, `persistence_image`

### 1. Compute Amplitude Vectors — ModelNet40

```bash
python tda_amplitude_project/generate_dataset_vectors.py --dataset modelnet40 --amplitude_metric bottleneck
```

### 2. Train PPTNet — Amplitude — ModelNet40

```bash
python tda_amplitude_project/train_pptnet_vectors.py --dataset modelnet40 --amplitude_metric bottleneck
```

### 3. Compute Amplitude Vectors — ScanObjectNN

```bash
python tda_amplitude_project/generate_dataset_vectors.py --dataset sonn --amplitude_metric bottleneck --n_jobs 14
```

### 4. Train PPTNet — Amplitude — ScanObjectNN

```bash
python tda_amplitude_project/train_pptnet_vectors.py --dataset sonn --amplitude_metric bottleneck --device cuda
```

### 5. Compute Entropy Vectors — ModelNet40

```bash
python tda_entropy_project/compute_entropy.py --dataset modelnet40 --n_jobs 14
```

### 6. Train PPTNet — Entropy — ModelNet40

```bash
python tda_entropy_project/train_pptnet_entropy.py --dataset modelnet40 --device cuda --batch_size 16 --num_workers 8
```

### 7. Compute Entropy Vectors — ScanObjectNN

```bash
python tda_entropy_project/compute_entropy.py --dataset sonn --n_jobs 14
```

### 8. Train PPTNet — Entropy — ScanObjectNN

```bash
python tda_entropy_project/train_pptnet_entropy.py --dataset sonn --device cuda --batch_size 16 --num_workers 8
```

### 9. Evaluate Structural Consistency

```bash
python evaluate_vector_structure.py
```

### 10. Plot Results

```bash
python plot.py
```
