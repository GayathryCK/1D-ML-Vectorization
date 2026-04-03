import argparse, hashlib, json, math, random, sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
if str(ROOT_DIR) not in sys.path: sys.path.insert(0, str(ROOT_DIR))

from pptnet_entropy_regressor import PPTNetEntropyRegressor

def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def stable_seed_from_string(s: str, base_seed: int = 42) -> int:
    h = hashlib.md5((s + str(base_seed)).encode("utf-8")).hexdigest()
    return int(h[:8], 16)

def load_point_cloud(txt_path: Path) -> np.ndarray:
    try: pts = np.loadtxt(txt_path, delimiter=",").astype(np.float32)
    except Exception: pts = np.loadtxt(txt_path).astype(np.float32)
    if pts.ndim == 1: pts = pts.reshape(1, -1)
    return pts[:, :3]

def sample_points(points: np.ndarray, num_points: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed); idx = rng.choice(points.shape[0], num_points, replace=False)
    return points[idx].astype(np.float32)

def normalize_points(points: np.ndarray) -> np.ndarray:
    points = points - np.mean(points, axis=0, keepdims=True)
    scale = np.max(np.linalg.norm(points, axis=1))
    if scale > 0: points = points / scale
    return points.astype(np.float32)

def ensure_txt_filename(sample_id: str) -> str:
    return sample_id if sample_id.lower().endswith(".txt") else f"{sample_id}.txt"

def parse_sample_id(sample_id: str, dataset: str):
    stem = Path(sample_id).stem
    parts = stem.split("_")
    if dataset == "modelnet40":
        class_name = "_".join(parts[:-1])
    elif dataset == "sonn":
        if len(parts) < 3:
            raise ValueError(f"Unexpected SONN sample_id format: {sample_id}")
        class_name = parts[1]
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return class_name

def rmse_from_mse(mse: float) -> float:
    return math.sqrt(mse)

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
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
    r2_mean = float(np.mean(r2_per_dim))

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
        "r2_H0": float(r2_per_dim[0]),
        "r2_H1": float(r2_per_dim[1]),
        "r2_H2": float(r2_per_dim[2]),
        "r2_mean": r2_mean,
    }


def get_model_params() -> dict:
    return {"SAMPLING": [1024, 256, 64, 16],
            "KNN": [32, 32, 16, 16],
            "FEATURE_SIZE": [64, 128, 256, 512],
            "GROUP": 4,
            "CLUSTER_SIZE": [64,64,64,64],
            "OUTPUT_DIM": [256,256,256,256],
            "GATING": True,
        }


class PointCloudEntropyDataset(Dataset):
    def __init__(self, csv_file, data_root, num_points=2048, normalize=False, seed=42, dataset="modelnet40"):
        self.df = pd.read_csv(csv_file)
        self.data_root = Path(data_root)
        self.num_points = num_points
        self.normalize = normalize
        self.seed = seed
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        sample_id = row["sample_id"]
        class_name = parse_sample_id(sample_id, self.dataset)
        txt_name = ensure_txt_filename(sample_id)
        txt_path = self.data_root / class_name / txt_name

        points = load_point_cloud(txt_path)
        s = stable_seed_from_string(sample_id, self.seed)
        points = sample_points(points, self.num_points, s)

        if self.normalize: points = normalize_points(points)

        ground_truth = np.array([row["H0"], row["H1"], row["H2"]], dtype=np.float32)

        return {
            "points": torch.from_numpy(points),
            "ground_truth": torch.from_numpy(ground_truth),
            "sample_id": sample_id,
            "class_name": class_name,
            "class_idx": int(row["class_idx"]),
        }


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for batch in tqdm(loader, desc="Training", leave=False):
        points = batch["points"].to(device, non_blocking=True)
        ground_truth = batch["ground_truth"].to(device, non_blocking=True)

        optimizer.zero_grad()
        pred = model(points)
        loss = criterion(pred, ground_truth)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * points.size(0)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_ground_truth, all_preds, all_sample_ids, all_class_names, all_class_idx = [], [], [], [], []

    for batch in tqdm(loader, desc="Training", leave=False):
        points = batch["points"].to(device)
        ground_truth = batch["ground_truth"].to(device)

        pred = model(points)
        loss = criterion(pred, ground_truth)
        running_loss += loss.item() * points.size(0)

        all_ground_truth.append(ground_truth.cpu().numpy())
        all_preds.append(pred.cpu().numpy())
        all_sample_ids.extend(batch["sample_id"])
        all_class_names.extend(batch["class_name"])
        all_class_idx.extend(batch["class_idx"].tolist())

    y_true = np.concatenate(all_ground_truth, axis=0)
    y_pred = np.concatenate(all_preds, axis=0)

    metrics = compute_metrics(y_true, y_pred)
    metrics["loss"] = running_loss / len(loader.dataset)

    pred_df = pd.DataFrame({
        "sample_id": all_sample_ids,
        "class_name": all_class_names,
        "class_idx": all_class_idx,
        "H0_true": y_true[:, 0],
        "H1_true": y_true[:, 1],
        "H2_true": y_true[:, 2],
        "H0_pred": y_pred[:, 0],
        "H1_pred": y_pred[:, 1],
        "H2_pred": y_pred[:, 2],
        "H0_abs_err": np.abs(y_pred[:, 0] - y_true[:, 0]),
        "H1_abs_err": np.abs(y_pred[:, 1] - y_true[:, 1]),
        "H2_abs_err": np.abs(y_pred[:, 2] - y_true[:, 2]),
    })

    return metrics, pred_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="modelnet40", choices=["modelnet40", "sonn"])
    parser.add_argument("--data_root", type=str, default="")
    parser.add_argument("--train_csv", type=str, default="")
    parser.add_argument("--test_csv", type=str, default="")
    parser.add_argument("--out_dir", type=str, default="")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_points", type=int, default=2048)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--normalize", action="store_true")
    args = parser.parse_args()

    cfg = {
        "modelnet40": {
            "data_root": r"D:\GRE\PPT-Net\data\ModelNet40",
            "train_csv": r"D:\GRE\PPT-Net\tda_entropy_project\outputs\modelnet40\ground_truths\mn40_train_entropy.csv",
            "test_csv": r"D:\GRE\PPT-Net\tda_entropy_project\outputs\modelnet40\ground_truths\mn40_test_entropy.csv",
            "out_dir": r"D:\GRE\PPT-Net\tda_entropy_project\outputs\modelnet40\train_entropy",
        },
        "sonn": {
            "data_root": r"D:\GRE\PPT-Net\data\SONNDataset\SONN",
            "train_csv": r"D:\GRE\PPT-Net\tda_entropy_project\outputs\sonn\ground_truths\sonn_train_entropy.csv",
            "test_csv": r"D:\GRE\PPT-Net\tda_entropy_project\outputs\sonn\ground_truths\sonn_test_entropy.csv",
            "out_dir": r"D:\GRE\PPT-Net\tda_entropy_project\outputs\sonn\train_entropy",
        },
    }

    c = cfg[args.dataset]

    if args.data_root == "":
        args.data_root = c["data_root"]
    if args.train_csv == "":
        args.train_csv = c["train_csv"]
    if args.test_csv == "":
        args.test_csv = c["test_csv"]
    if args.out_dir == "":
        args.out_dir = c["out_dir"]

    out_dir = Path(args.out_dir)/args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if args.device.lower() == "cuda" and torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("PPTNet Entropy Training")
    print("=" * 70)
    print(f"dataset       : {args.dataset}")
    print(f"data_root     : {args.data_root}")
    print(f"train_csv     : {args.train_csv}")
    print(f"test_csv      : {args.test_csv}")
    print(f"out_dir       : {args.out_dir}")
    print(f"epochs        : {args.epochs}")
    print(f"batch_size    : {args.batch_size}")
    print(f"lr            : {args.lr}")
    print(f"weight_decay  : {args.weight_decay}")
    print(f"num_points    : {args.num_points}")
    print(f"num_workers   : {args.num_workers}")
    print(f"normalize     : {args.normalize}")
    print(f"device        : {device}")
    print("=" * 70)

    train_ds = PointCloudEntropyDataset(csv_file=Path(args.train_csv), data_root=Path(args.data_root), num_points=args.num_points, normalize=args.normalize, seed=args.seed, dataset=args.dataset)
    test_ds = PointCloudEntropyDataset(csv_file=Path(args.test_csv), data_root=Path(args.data_root), num_points=args.num_points, normalize=args.normalize, seed=args.seed, dataset=args.dataset)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=(device.type == "cuda"), drop_last=False, persistent_workers=True, prefetch_factor=4)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == "cuda"), drop_last=False, persistent_workers=True, prefetch_factor=4)

    model = PPTNetEntropyRegressor(get_model_params()).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_test_mae = float("inf")
    best_epoch = -1
    patience = 5
    patience_counter = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        test_metrics, test_pred_df = evaluate(model, test_loader, criterion, device)

        row = {"epoch": epoch, "train_loss": train_loss, "test_loss": test_metrics["loss"], "test_mae": test_metrics["mae"], "test_rmse": test_metrics["rmse"], "test_r2_mean": test_metrics["r2_mean"], "mae_H0": test_metrics["mae_H0"], "mae_H1": test_metrics["mae_H1"], "mae_H2": test_metrics["mae_H2"]}
        history.append(row)

        print(f"Epoch [{epoch:03d}/{args.epochs:03d}]  train_loss={train_loss:.6f}  test_loss={test_metrics['loss']:.6f}  test_mae={test_metrics['mae']:.6f}  test_rmse={test_metrics['rmse']:.6f}  r2_mean={test_metrics['r2_mean']:.6f}")

        if test_metrics["mae"] < best_test_mae:
            best_test_mae = test_metrics["mae"]
            best_epoch = epoch
            patience_counter = 0

            ckpt_path = out_dir / "pptnet_entropy_best.pth"
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                 "test_metrics": test_metrics, "args": vars(args), "model_params": get_model_params()}, ckpt_path)

            pred_path = out_dir / "best_test_predictions.csv"
            test_pred_df.to_csv(pred_path, index=False)

            metrics_path = out_dir / "best_test_metrics.json"
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump({"best_epoch": epoch, **test_metrics}, f, indent=2)

            print(f"  [BEST] saved checkpoint, predictions, and metrics at epoch {epoch}")

        # else:
        #     patience_counter += 1
        #     print(f"  No improvement for {patience_counter}/{patience} epoch(s)")
        #
        #     if patience_counter >= patience:
        #         print(f"\nEarly stopping triggered at epoch {epoch}")
        #         break

    history_df = pd.DataFrame(history)
    history_df.to_csv(out_dir / "training_history.csv", index=False)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best epoch      : {best_epoch}")
    print(f"Best test MAE   : {best_test_mae:.6f}")
    print(f"Checkpoint      : {out_dir / 'pptnet_entropy_best.pth'}")
    print(f"Predictions CSV : {out_dir / 'best_test_predictions.csv'}")
    print(f"Metrics JSON    : {out_dir / 'best_test_metrics.json'}")
    print(f"History CSV     : {out_dir / 'training_history.csv'}")
    print("=" * 70)


if __name__ == "__main__":
    main()