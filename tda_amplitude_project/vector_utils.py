import hashlib
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stable_seed_from_string(s: str, base_seed: int = 42) -> int:
    h = hashlib.md5((s + str(base_seed)).encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def load_split_ids(split_file):
    split_file = Path(split_file)
    with open(split_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_point_cloud(txt_path: Path) -> np.ndarray:
    try:
        pts = np.loadtxt(txt_path, delimiter=",").astype(np.float32)
    except Exception:
        pts = np.loadtxt(txt_path).astype(np.float32)

    if pts.ndim == 1:
        pts = pts.reshape(1, -1)

    return pts[:, :3]


def sample_points(points: np.ndarray, num_points: int, seed: int) -> np.ndarray:
    if points.shape[0] == num_points:
        return points.astype(np.float32)

    rng = np.random.default_rng(seed)

    if points.shape[0] > num_points:
        idx = rng.choice(points.shape[0], num_points, replace=False)
    else:
        idx = rng.choice(points.shape[0], num_points, replace=True)

    return points[idx].astype(np.float32)


def normalize_points(points: np.ndarray) -> np.ndarray:
    points = points - np.mean(points, axis=0, keepdims=True)
    scale = np.max(np.linalg.norm(points, axis=1))
    if scale > 0:
        points = points / scale
    return points.astype(np.float32)


def infer_target_columns(df: pd.DataFrame):
    cols = [c for c in df.columns if c.startswith("f_")]
    cols = sorted(cols, key=lambda x: int(x.split("_")[1]))
    if not cols:
        raise ValueError("No target columns found. Expected columns like f_0, f_1, ...")
    return cols


def rmse_from_mse(mse: float) -> float:
    return math.sqrt(mse)


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    diff = y_pred - y_true

    mse_all = float(np.mean(diff ** 2))
    mae_all = float(np.mean(np.abs(diff)))
    rmse_all = rmse_from_mse(mse_all)

    per_dim_mse = np.mean(diff ** 2, axis=0)
    per_dim_mae = np.mean(np.abs(diff), axis=0)
    per_dim_rmse = np.sqrt(per_dim_mse)

    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2, axis=0)
    r2_per_dim = np.where(ss_tot > 0, 1.0 - (ss_res / ss_tot), 0.0)

    metrics = {
        "mse": mse_all,
        "mae": mae_all,
        "rmse": rmse_all,
        "mae_H0": float(per_dim_mae[0]),
        "mae_H1": float(per_dim_mae[1]),
        "mae_H2": float(per_dim_mae[2]),
        "rmse_H0": float(per_dim_rmse[0]),
        "rmse_H1": float(per_dim_rmse[1]),
        "rmse_H2": float(per_dim_rmse[2]),
        "r2_H0": float(r2_per_dim[0]),
        "r2_H1": float(r2_per_dim[1]),
        "r2_H2": float(r2_per_dim[2]),
        "r2_mean": float(np.mean(r2_per_dim)),
    }

    return metrics


def get_model_params() -> dict:
    return {
        "SAMPLING": [1024, 256, 64, 16],
        "KNN": [32, 32, 16, 16],
        "FEATURE_SIZE": [64, 128, 256, 512],
        "GROUP": 4,
        "CLUSTER_SIZE": [64, 64, 64, 64],
        "OUTPUT_DIM": [256, 256, 256, 256],
        "GATING": True,
    }