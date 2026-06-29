"""
predict_ripsnet_entropy.py

Pass ModelNet40 test point clouds through a saved RipsNet entropy model and
produce a predictions CSV with columns:
  sample_id, class_name, class_idx,
  H0_true, H1_true, H2_true,
  H0_pred, H1_pred, H2_pred,
  H0_abs_err, H1_abs_err, H2_abs_err

Usage example:
  python predict_ripsnet_entropy.py \
      --weights_path  ripsnet_entropy_out/ripsnet_entropy_best.weights.h5 \
      --test_csv      ground_truths/mn40_test_entropy.csv \
      --data_root     ModelNet40 \
      --out_csv       ripsnet_entropy_out/ripsnet_test_predictions.csv
"""

import argparse
import hashlib
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
RIPSNET_DIR = SCRIPT_DIR / "ripsnet-main"
if str(RIPSNET_DIR) not in sys.path:
    sys.path.insert(0, str(RIPSNET_DIR))

from utils import DenseRagged, PermopRagged  # noqa: E402


# ── mirror of the model from train_ripsnet_entropy.py ─────────────────────────
def build_ripsnet_entropy_model() -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(None, 3), dtype="float32", ragged=True)
    x = DenseRagged(units=64, use_bias=True, activation="relu")(inputs)
    x = DenseRagged(units=64, use_bias=True, activation="relu")(x)
    x = DenseRagged(units=32, use_bias=True, activation="relu")(x)
    x = PermopRagged()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dense(64,  activation="relu")(x)
    outputs = tf.keras.layers.Dense(3, activation="linear")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="ripsnet_entropy")


# ── helpers (duplicated from train script for standalone use) ──────────────────
def stable_seed_from_string(s: str, base_seed: int = 42) -> int:
    h = hashlib.md5((s + str(base_seed)).encode()).hexdigest()
    return int(h[:8], 16) % (2**31)


def load_point_cloud(txt_path: Path) -> np.ndarray:
    try:
        pts = np.loadtxt(txt_path, delimiter=",").astype(np.float32)
    except Exception:
        pts = np.loadtxt(txt_path).astype(np.float32)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    return pts[:, :3]


def sample_points(points: np.ndarray, num_points: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    replace = points.shape[0] < num_points
    idx = rng.choice(points.shape[0], num_points, replace=replace)
    return points[idx].astype(np.float32)


def normalize_points(points: np.ndarray) -> np.ndarray:
    points = points - np.mean(points, axis=0, keepdims=True)
    scale = np.max(np.linalg.norm(points, axis=1))
    if scale > 0:
        points = points / scale
    return points.astype(np.float32)


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Predict entropy with saved RipsNet")
    parser.add_argument("--weights_path", type=str, default="",
                        help="Path to .weights.h5 file saved during training")
    parser.add_argument("--test_csv",   type=str, default="",
                        help="Ground-truth CSV for the test split")
    parser.add_argument("--data_root",  type=str, default="",
                        help="ModelNet40 root directory")
    parser.add_argument("--out_csv",    type=str, default="",
                        help="Where to write the predictions CSV")
    parser.add_argument("--num_points", type=int, default=1024,
                        help="Points to sample per cloud (must match training)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--normalize",  action="store_true")
    args = parser.parse_args()

    # Defaults
    if not args.weights_path:
        args.weights_path = str(
            SCRIPT_DIR / "ripsnet_entropy_out" / "ripsnet_entropy_best.weights.h5")
    if not args.test_csv:
        args.test_csv = str(SCRIPT_DIR / "ground_truths" / "mn40_test_entropy.csv")
    if not args.data_root:
        args.data_root = str(SCRIPT_DIR / "ModelNet40")
    if not args.out_csv:
        args.out_csv = str(
            SCRIPT_DIR / "ripsnet_entropy_out" / "ripsnet_test_predictions.csv")

    print("=" * 70)
    print("RipsNet Entropy Regressor — Prediction")
    print("=" * 70)
    print(f"  weights    : {args.weights_path}")
    print(f"  test_csv   : {args.test_csv}")
    print(f"  data_root  : {args.data_root}")
    print(f"  out_csv    : {args.out_csv}")
    print(f"  num_points : {args.num_points}")
    print("=" * 70)

    # ── rebuild model and load weights ─────────────────────────────────────────
    # Keras 3 requires get_config() on custom layers for full model
    # serialisation; we avoid that by rebuilding the architecture and loading
    # weights only — keeping ripsnet-main/utils.py untouched.
    print(f"\nBuilding model and loading weights from {args.weights_path} …")
    model = build_ripsnet_entropy_model()
    # Warm up the model with one dummy forward pass so weights can be loaded
    dummy = tf.RaggedTensor.from_tensor(
        tf.zeros([1, args.num_points, 3]), ragged_rank=1)
    model(dummy, training=False)
    model.load_weights(args.weights_path)
    print("Weights loaded.")

    # ── load test point clouds ─────────────────────────────────────────────────
    df = pd.read_csv(args.test_csv)
    n = len(df)
    clouds_np = np.empty((n, args.num_points, 3), dtype=np.float32)

    for i, (_, row) in enumerate(tqdm(df.iterrows(), total=n,
                                      desc="Loading test clouds")):
        sid = row["sample_id"]
        cname = row["class_name"]
        txt_path = Path(args.data_root) / cname / f"{sid}.txt"

        pts = load_point_cloud(txt_path)
        s = stable_seed_from_string(sid, args.seed)
        pts = sample_points(pts, args.num_points, s)
        if args.normalize:
            pts = normalize_points(pts)
        clouds_np[i] = pts

    # ── run inference (per-batch numpy → RaggedTensor conversion) ─────────────
    print("\nRunning inference …")
    preds = []
    for b in tqdm(range(math.ceil(n / args.batch_size)), desc="Batches"):
        batch_np = clouds_np[b * args.batch_size: (b + 1) * args.batch_size]
        batch_ragged = tf.RaggedTensor.from_tensor(
            tf.constant(batch_np), ragged_rank=1)
        out = model(batch_ragged, training=False)
        preds.append(out.numpy())

    preds = np.vstack(preds)

    # ── save predictions ───────────────────────────────────────────────────────
    true_H0 = df["H0"].values.astype(np.float32)
    true_H1 = df["H1"].values.astype(np.float32)
    true_H2 = df["H2"].values.astype(np.float32)

    out_df = pd.DataFrame({
        "sample_id":  df["sample_id"].tolist(),
        "class_name": df["class_name"].tolist(),
        "class_idx":  df["class_idx"].tolist(),
        "H0_true": true_H0,
        "H1_true": true_H1,
        "H2_true": true_H2,
        "H0_pred": preds[:, 0],
        "H1_pred": preds[:, 1],
        "H2_pred": preds[:, 2],
        "H0_abs_err": np.abs(preds[:, 0] - true_H0),
        "H1_abs_err": np.abs(preds[:, 1] - true_H1),
        "H2_abs_err": np.abs(preds[:, 2] - true_H2),
    })

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print(f"\nPredictions saved to {out_path}")
    print(f"  Rows: {len(out_df)}")


if __name__ == "__main__":
    main()
