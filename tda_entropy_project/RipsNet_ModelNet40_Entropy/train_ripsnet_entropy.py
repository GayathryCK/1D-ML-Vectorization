"""
train_ripsnet_entropy.py

Train a RipsNet-style DeepSet model on ModelNet40 point clouds to predict
persistent-homology entropy vectors [H0, H1, H2].

Architecture (mirrors the RipsNet paper, adapted for 3-D regression):
  DenseRagged(64) → DenseRagged(64) → DenseRagged(32)
  → PermopRagged (sum-pool to fixed vector)
  → Dense(256) → Dense(128) → Dense(64) → Dense(3, linear)

Inputs  : ModelNet40 .txt point clouds  (subsampled to --num_points)
Targets : ground-truth entropy CSV      (H0, H1, H2 columns)
Output  : SavedModel in <out_dir>/ripsnet_entropy_best/
          + best_test_predictions.csv, best_test_metrics.json,
            training_history.csv
"""

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

# ── resolve ripsnet-main so we can import DenseRagged / PermopRagged ──────────
SCRIPT_DIR = Path(__file__).resolve().parent
RIPSNET_DIR = SCRIPT_DIR / "ripsnet-main"
if str(RIPSNET_DIR) not in sys.path:
    sys.path.insert(0, str(RIPSNET_DIR))

from utils import DenseRagged, PermopRagged  # noqa: E402


# ── reproducibility ────────────────────────────────────────────────────────────
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def stable_seed_from_string(s: str, base_seed: int = 42) -> int:
    h = hashlib.md5((s + str(base_seed)).encode()).hexdigest()
    return int(h[:8], 16) % (2**31)


# ── point-cloud I/O ────────────────────────────────────────────────────────────
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


# ── metrics ────────────────────────────────────────────────────────────────────
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    diff = y_pred - y_true
    mse_all = float(np.mean(diff ** 2))
    mae_all = float(np.mean(np.abs(diff)))
    rmse_all = math.sqrt(mse_all)

    per_dim_mae = np.mean(np.abs(diff), axis=0)
    per_dim_rmse = np.sqrt(np.mean(diff ** 2, axis=0))

    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2, axis=0)
    r2_per = np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, 0.0)

    return {
        "mse": mse_all,
        "mae": mae_all,
        "rmse": rmse_all,
        "mae_H0": float(per_dim_mae[0]),
        "mae_H1": float(per_dim_mae[1]),
        "mae_H2": float(per_dim_mae[2]),
        "rmse_H0": float(per_dim_rmse[0]),
        "rmse_H1": float(per_dim_rmse[1]),
        "rmse_H2": float(per_dim_rmse[2]),
        "r2_H0": float(r2_per[0]),
        "r2_H1": float(r2_per[1]),
        "r2_H2": float(r2_per[2]),
        "r2_mean": float(np.mean(r2_per)),
    }


# ── data loading ───────────────────────────────────────────────────────────────
def load_dataset(csv_file: Path, data_root: Path, num_points: int,
                 normalize: bool, seed: int):
    """Returns (clouds_np [N, num_points, 3], labels_np, sample_ids, class_names, class_idxs).

    Storing as a plain numpy array avoids the prohibitively slow
    tf.ragged.constant() call over ~10k point clouds. Each batch is converted
    to a RaggedTensor on-the-fly in the training loop (see numpy_batch_to_ragged).
    """
    df = pd.read_csv(csv_file)
    n = len(df)
    clouds_np = np.empty((n, num_points, 3), dtype=np.float32)
    labels    = np.empty((n, 3), dtype=np.float32)
    sample_ids, class_names, class_idxs = [], [], []

    for i, (_, row) in enumerate(tqdm(df.iterrows(), total=n,
                                      desc=f"Loading {csv_file.name}")):
        sid   = row["sample_id"]
        cname = row["class_name"]
        txt_path = data_root / cname / (sid if sid.endswith(".txt") else f"{sid}.txt")

        pts = load_point_cloud(txt_path)
        s   = stable_seed_from_string(sid, seed)
        pts = sample_points(pts, num_points, s)
        if normalize:
            pts = normalize_points(pts)

        clouds_np[i] = pts
        labels[i]    = [float(row["H0"]), float(row["H1"]), float(row["H2"])]
        sample_ids.append(sid)
        class_names.append(cname)
        class_idxs.append(int(row["class_idx"]))

    return clouds_np, labels, sample_ids, class_names, class_idxs


def numpy_batch_to_ragged(batch_np: np.ndarray) -> tf.RaggedTensor:
    """Convert a (B, N, 3) numpy batch to a (B, None, 3) RaggedTensor.

    tf.RaggedTensor.from_tensor is O(B) — only the row_splits array is built,
    no data is copied. This keeps the authentic DenseRagged/PermopRagged
    forward pass while being orders of magnitude faster than building a
    whole-dataset ragged tensor upfront.
    """
    return tf.RaggedTensor.from_tensor(tf.constant(batch_np), ragged_rank=1)


# ── model ──────────────────────────────────────────────────────────────────────
def build_ripsnet_entropy_model() -> tf.keras.Model:
    """
    RipsNet DeepSet architecture for 3-D point-cloud → [H0, H1, H2] regression.

    Pointwise branch (DenseRagged):  (N, 3) → (N, 64) → (N, 64) → (N, 32)
    Aggregation  (PermopRagged):     (N, 32) → (32,)   [sum pool]
    Regression head (Dense):         32 → 256 → 128 → 64 → 3
    """
    inputs = tf.keras.Input(shape=(None, 3), dtype="float32", ragged=True)

    # Pointwise feature extraction
    x = DenseRagged(units=64, use_bias=True, activation="relu")(inputs)
    x = DenseRagged(units=64, use_bias=True, activation="relu")(x)
    x = DenseRagged(units=32, use_bias=True, activation="relu")(x)

    # Permutation-invariant aggregation
    x = PermopRagged()(x)   # → fixed-size vector of dim 32

    # Regression head
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dense(64,  activation="relu")(x)
    outputs = tf.keras.layers.Dense(3, activation="linear")(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="ripsnet_entropy")


# ── prediction helper ──────────────────────────────────────────────────────────
def predict_in_batches(model: tf.keras.Model, clouds_np: np.ndarray,
                       batch_size: int) -> np.ndarray:
    n = clouds_np.shape[0]
    preds = []
    for b in range(math.ceil(n / batch_size)):
        batch_ragged = numpy_batch_to_ragged(
            clouds_np[b * batch_size: (b + 1) * batch_size])
        out = model(batch_ragged, training=False)
        preds.append(out.numpy())
    return np.vstack(preds)


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train RipsNet entropy regressor")
    parser.add_argument("--data_root",  type=str, default="")
    parser.add_argument("--train_csv",  type=str, default="")
    parser.add_argument("--test_csv",   type=str, default="")
    parser.add_argument("--out_dir",    type=str, default="")
    parser.add_argument("--epochs",     type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr",         type=float, default=5e-4)
    parser.add_argument("--num_points", type=int, default=1024)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--normalize",  action="store_true")
    args = parser.parse_args()

    # ── defaults (works both locally and on EC2 after rsync) ──────────────────
    if not args.data_root:
        args.data_root = str(SCRIPT_DIR / "ModelNet40")
    if not args.train_csv:
        args.train_csv = str(SCRIPT_DIR / "ground_truths" / "mn40_train_entropy.csv")
    if not args.test_csv:
        args.test_csv  = str(SCRIPT_DIR / "ground_truths" / "mn40_test_entropy.csv")
    if not args.out_dir:
        args.out_dir   = str(SCRIPT_DIR / "ripsnet_entropy_out")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    print("=" * 70)
    print("RipsNet Entropy Regressor — Training")
    print("=" * 70)
    for k, v in vars(args).items():
        print(f"  {k:<14}: {v}")
    print("=" * 70)

    # ── load data ──────────────────────────────────────────────────────────────
    print("\nLoading training data …")
    train_clouds, train_labels, train_ids, train_cls, train_idx = load_dataset(
        Path(args.train_csv), Path(args.data_root),
        args.num_points, args.normalize, args.seed)
    print(f"  train clouds shape: {train_clouds.shape}")

    print("\nLoading test data …")
    test_clouds, test_labels, test_ids, test_cls, test_idx = load_dataset(
        Path(args.test_csv), Path(args.data_root),
        args.num_points, args.normalize, args.seed)
    print(f"  test  clouds shape: {test_clouds.shape}")

    n_train = train_labels.shape[0]

    # ── build & compile ────────────────────────────────────────────────────────
    model = build_ripsnet_entropy_model()
    model.summary()

    optimizer = tf.keras.optimizers.Adamax(learning_rate=args.lr)

    # ── training loop ──────────────────────────────────────────────────────────
    best_mae = float("inf")
    best_epoch = -1
    history = []

    for epoch in range(1, args.epochs + 1):
        # Shuffle
        perm = np.random.permutation(n_train)
        epoch_loss = 0.0
        n_batches = math.ceil(n_train / args.batch_size)

        for b in tqdm(range(n_batches), desc=f"Epoch {epoch:03d}/{args.epochs:03d}",
                      leave=False):
            batch_idx    = perm[b * args.batch_size: (b + 1) * args.batch_size]
            batch_clouds = numpy_batch_to_ragged(train_clouds[batch_idx])
            batch_labels = train_labels[batch_idx]

            with tf.GradientTape() as tape:
                preds = model(batch_clouds, training=True)
                loss = tf.reduce_mean(tf.square(preds - batch_labels))

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            epoch_loss += loss.numpy() * len(batch_idx)

        train_loss = epoch_loss / n_train

        # Evaluate on test set
        test_preds = predict_in_batches(model, test_clouds, args.batch_size)
        metrics = compute_metrics(test_labels, test_preds)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "test_loss": metrics["mse"],
            "test_mae": metrics["mae"],
            "test_rmse": metrics["rmse"],
            "test_r2_mean": metrics["r2_mean"],
            "mae_H0": metrics["mae_H0"],
            "mae_H1": metrics["mae_H1"],
            "mae_H2": metrics["mae_H2"],
        }
        history.append(row)

        print(
            f"Epoch [{epoch:03d}/{args.epochs:03d}]  "
            f"train_loss={train_loss:.6f}  "
            f"test_mae={metrics['mae']:.6f}  "
            f"test_rmse={metrics['rmse']:.6f}  "
            f"r2_mean={metrics['r2_mean']:.4f}  "
            f"[H0={metrics['mae_H0']:.4f} H1={metrics['mae_H1']:.4f} H2={metrics['mae_H2']:.4f}]"
        )

        # Save best
        if metrics["mae"] < best_mae:
            best_mae = metrics["mae"]
            best_epoch = epoch

            # Save weights (.h5).  Keras 3 requires get_config() on custom
            # layers for full model serialisation; saving weights avoids that
            # constraint while keeping utils.py untouched.
            weights_path = str(out_dir / "ripsnet_entropy_best.weights.h5")
            model.save_weights(weights_path)

            # Record the training config so predict_ripsnet_entropy.py can
            # rebuild the identical architecture before loading weights.
            train_cfg = {"num_points": args.num_points,
                         "architecture": "DenseRagged(64)-DenseRagged(64)-DenseRagged(32)-PermopRagged-Dense(256)-Dense(128)-Dense(64)-Dense(3)"}
            with open(out_dir / "ripsnet_train_config.json", "w") as f:
                json.dump(train_cfg, f, indent=2)

            pred_df = pd.DataFrame({
                "sample_id":  test_ids,
                "class_name": test_cls,
                "class_idx":  test_idx,
                "H0_true": test_labels[:, 0],
                "H1_true": test_labels[:, 1],
                "H2_true": test_labels[:, 2],
                "H0_pred": test_preds[:, 0],
                "H1_pred": test_preds[:, 1],
                "H2_pred": test_preds[:, 2],
                "H0_abs_err": np.abs(test_preds[:, 0] - test_labels[:, 0]),
                "H1_abs_err": np.abs(test_preds[:, 1] - test_labels[:, 1]),
                "H2_abs_err": np.abs(test_preds[:, 2] - test_labels[:, 2]),
            })
            pred_df.to_csv(out_dir / "best_test_predictions.csv", index=False)

            with open(out_dir / "best_test_metrics.json", "w") as f:
                json.dump({"best_epoch": epoch, **metrics}, f, indent=2)

            print(f"  [BEST] model, predictions, and metrics saved (epoch {epoch})")

    # Save full history
    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Best epoch  : {best_epoch}")
    print(f"  Best MAE    : {best_mae:.6f}")
    print(f"  Weights     : {out_dir / 'ripsnet_entropy_best.weights.h5'}")
    print(f"  Predictions : {out_dir / 'best_test_predictions.csv'}")
    print(f"  Metrics     : {out_dir / 'best_test_metrics.json'}")
    print(f"  History     : {out_dir / 'training_history.csv'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
