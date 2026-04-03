import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pptnet_vector_regressor import PPTNetVectorRegressor
from vector_utils import (
    compute_regression_metrics,
    get_model_params,
    load_point_cloud,
    normalize_points,
    sample_points,
    set_seed,
    stable_seed_from_string,
)

DATASET_CONFIG = {
    "modelnet40": {
        "data_root": r"D:\GRE\PPT-Net\data\ModelNet40",
        "out_root": r"D:\GRE\PPT-Net\tda_vector_project\outputs",
        "prefix": "mn40",
        "title": "ModelNet40",
    },
    "sonn": {
        "data_root": r"D:\GRE\PPT-Net\data\SONNDataSet\SONN",
        "out_root": r"D:\GRE\PPT-Net\tda_vector_project\SONN_outputs",
        "prefix": "sonn",
        "title": "SONN",
    },
}


def get_dataset_config(dataset_name: str):
    dataset_name = dataset_name.lower().strip()
    if dataset_name not in DATASET_CONFIG:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Choose from: {list(DATASET_CONFIG.keys())}")
    return DATASET_CONFIG[dataset_name]


def ensure_txt_filename(sample_id: str) -> str:
    return sample_id if sample_id.lower().endswith(".txt") else f"{sample_id}.txt"

class PointCloudVectorDataset(Dataset):
    def __init__(
        self,
        csv_file: Path,
        data_root: Path,
        num_points: int = 2048,
        normalize: bool = False,
        seed: int = 42,
    ):
        self.df = pd.read_csv(csv_file)
        self.data_root = Path(data_root)
        self.num_points = num_points
        self.normalize = normalize
        self.seed = seed

        self.feature_cols = ["H0", "H1", "H2"]
        self.target_dim = 3

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        sample_id = row["sample_id"]
        class_name = row["class_name"]
        class_idx = int(row["class_idx"])

        txt_name = ensure_txt_filename(sample_id)
        txt_path = self.data_root / class_name / txt_name
        if not txt_path.exists():
            raise FileNotFoundError(f"Point cloud file not found: {txt_path}")
        points = load_point_cloud(txt_path)

        seed = stable_seed_from_string(sample_id, self.seed)
        points = sample_points(points, self.num_points, seed)

        if self.normalize:
            points = normalize_points(points)

        target = row[self.feature_cols].to_numpy(dtype=np.float32)

        points = torch.from_numpy(points).float()
        target = torch.from_numpy(target).float()

        return points, target, sample_id, class_name, class_idx


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for points, targets, _, _, _ in tqdm(loader, desc="Training", leave=False):
        points = points.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        preds = model(points)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * points.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_sample_ids = []
    all_class_names = []
    all_class_idxs = []

    for points, targets, sample_ids, class_names, class_idxs in tqdm(loader, desc="Evaluating", leave=False):
        points = points.to(device)
        targets = targets.to(device)

        preds = model(points)
        loss = criterion(preds, targets)
        total_loss += loss.item() * points.size(0)

        all_preds.append(preds.detach().cpu().numpy())
        all_targets.append(targets.detach().cpu().numpy())
        all_sample_ids.extend(sample_ids)
        all_class_names.extend(class_names)

        if torch.is_tensor(class_idxs):
            all_class_idxs.extend(class_idxs.cpu().numpy().tolist())
        else:
            all_class_idxs.extend(list(class_idxs))

    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_targets, axis=0)

    metrics = compute_regression_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / len(loader.dataset)

    return metrics, y_true, y_pred, all_sample_ids, all_class_names, all_class_idxs


def save_predictions_csv(out_path: Path, sample_ids, class_names, class_idxs, y_true, y_pred):
    rows = []

    target_dim = y_true.shape[1]
    if target_dim != 3:
        raise ValueError(f"Expected target_dim=3 for H0/H1/H2 saving, got {target_dim}")

    for i, sid in enumerate(sample_ids):
        row = {
            "sample_id": sid,
            "class_name": class_names[i],
            "class_idx": int(class_idxs[i]),
            "H0_true": float(y_true[i, 0]),
            "H1_true": float(y_true[i, 1]),
            "H2_true": float(y_true[i, 2]),
            "H0_pred": float(y_pred[i, 0]),
            "H1_pred": float(y_pred[i, 1]),
            "H2_pred": float(y_pred[i, 2]),
            "H0_abs_err": float(abs(y_pred[i, 0] - y_true[i, 0])),
            "H1_abs_err": float(abs(y_pred[i, 1] - y_true[i, 1])),
            "H2_abs_err": float(abs(y_pred[i, 2] - y_true[i, 2])),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="modelnet40", choices=["modelnet40", "sonn"])
    parser.add_argument("--data_root", type=str, default="")
    parser.add_argument("--train_csv", type=str, default="")
    parser.add_argument("--test_csv", type=str, default="")
    parser.add_argument("--out_dir", type=str, default="")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_points", type=int, default=2048)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--amplitude_metric",
        type=str,
        default="bottleneck",
        choices=["bottleneck", "wasserstein", "betti", "landscape", "silhouette", "heat", "persistence_image"]
    )
    args = parser.parse_args()

    cfg = get_dataset_config(args.dataset)

    if args.data_root == "":
        args.data_root = cfg["data_root"]

    metric_root = Path(cfg["out_root"]) / f"amplitude_{args.amplitude_metric}"

    if args.train_csv == "":
        args.train_csv = str(metric_root / "ground_truths" / f"{cfg['prefix']}_train_amplitude_{args.amplitude_metric}.csv")

    if args.test_csv == "":
        args.test_csv = str(metric_root / "ground_truths" / f"{cfg['prefix']}_test_amplitude_{args.amplitude_metric}.csv")

    if args.out_dir == "":
        args.out_dir = str(metric_root / "train_results")

    set_seed(args.seed)

    data_root = Path(args.data_root)
    train_csv = Path(args.train_csv)
    test_csv = Path(args.test_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if (args.device.lower() == "cuda" and torch.cuda.is_available()) else "cpu")

    train_ds = PointCloudVectorDataset(
        csv_file=train_csv,
        data_root=data_root,
        num_points=args.num_points,
        normalize=args.normalize,
        seed=args.seed
    )
    test_ds = PointCloudVectorDataset(
        csv_file=test_csv,
        data_root=data_root,
        num_points=args.num_points,
        normalize=args.normalize,
        seed=args.seed
    )

    if train_ds.target_dim != test_ds.target_dim:
        raise ValueError(f"Train target_dim ({train_ds.target_dim}) != Test target_dim ({test_ds.target_dim})")

    target_dim = train_ds.target_dim

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = PPTNetVectorRegressor(param=get_model_params(), output_dim=target_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    best_rmse = float("inf")
    history = []

    print("=" * 70)
    print(f"PPTNet Vector Training - {cfg['title']}")
    print("=" * 70)
    print(f"dataset       : {args.dataset}")
    print(f"data_root     : {data_root}")
    print(f"train_csv     : {train_csv}")
    print(f"test_csv      : {test_csv}")
    print(f"out_dir       : {out_dir}")
    print(f"epochs        : {args.epochs}")
    print(f"batch_size    : {args.batch_size}")
    print(f"lr            : {args.lr}")
    print(f"weight_decay  : {args.weight_decay}")
    print(f"num_points    : {args.num_points}")
    print(f"num_workers   : {args.num_workers}")
    print(f"normalize     : {args.normalize}")
    print(f"device        : {device}")
    print(f"target_dim    : {target_dim}")
    print(f"train_size    : {len(train_ds)}")
    print(f"test_size     : {len(test_ds)}")
    print("=" * 70)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        test_metrics, y_true, y_pred, sample_ids, class_names, class_idxs = evaluate(
            model, test_loader, criterion, device
        )

        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "test_loss": float(test_metrics["loss"]),
            "test_mae": float(test_metrics["mae"]),
            "test_rmse": float(test_metrics["rmse"]),
            "r2_mean": float(test_metrics["r2_mean"]),
        }

        for k, v in test_metrics.items():
            if k not in row and isinstance(v, (float, int)):
                row[k] = float(v)

        history.append(row)

        print(
            f"Epoch [{epoch:03d}/{args.epochs}]  "
            f"train_loss={train_loss:.6f}  "
            f"test_loss={test_metrics['loss']:.6f}  "
            f"test_mae={test_metrics['mae']:.6f}  "
            f"test_rmse={test_metrics['rmse']:.6f}  "
            f"r2_mean={test_metrics['r2_mean']:.6f}"
        )

        if test_metrics["rmse"] < best_rmse:
            best_rmse = test_metrics["rmse"]

            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_rmse": best_rmse,
                "target_dim": target_dim,
                "args": vars(args),
            }
            torch.save(ckpt, out_dir / "best_model.pth")

            save_predictions_csv(
                out_dir / "best_test_predictions.csv",
                sample_ids,
                class_names,
                class_idxs,
                y_true,
                y_pred,
            )

            best_metrics = {
                "best_epoch": epoch,
                **test_metrics,
                "target_dim": target_dim,
            }
            with open(out_dir / "best_metrics.json", "w", encoding="utf-8") as f:
                json.dump(best_metrics, f, indent=2)

            print(
                f"[BEST] epoch={epoch:03d}  "
                f"rmse={test_metrics['rmse']:.6f}  "
                f"mae={test_metrics['mae']:.6f}  "
                f"saved to: {out_dir}"
            )

    history_df = pd.DataFrame(history)
    history_df.to_csv(out_dir / "training_history.csv", index=False)

    print("\nDone.")
    print(f"Best model saved to: {out_dir / 'best_model.pth'}")
    print(f"History saved to   : {out_dir / 'training_history.csv'}")
    print(f"Best metrics saved : {out_dir / 'best_metrics.json'}")
    print(f"Best preds saved   : {out_dir / 'best_test_predictions.csv'}")


if __name__ == "__main__":
    main()