import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

#from gtda.homology import VietorisRipsPersistence
from gtda.homology import WeakAlphaPersistence
from gtda.diagrams import PersistenceEntropy

def load_split_ids(split_file):
    with open(split_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_point_cloud(txt_path):
    try:
        pts = np.loadtxt(txt_path, delimiter=",").astype(np.float32)
    except Exception:
        pts = np.loadtxt(txt_path).astype(np.float32)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    return pts[:, :3]

def fix_num_points(points, num_points=2048):
    idx = np.random.choice(points.shape[0], num_points, replace=False)
    return points[idx]

def normalize_points(points):
    points = points - np.mean(points, axis=0, keepdims=True)
    scale = np.max(np.linalg.norm(points, axis=1))
    if scale > 0:
        points = points / scale
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

def compute_entropy(points):
    points = np.unique(points, axis=0)
    #vr = VietorisRipsPersistence(metric="euclidean", homology_dimensions=[0, 1, 2], collapse_edges=False, n_jobs=1)
    wap = WeakAlphaPersistence(homology_dimensions=[0, 1, 2], n_jobs=1)
    pe = PersistenceEntropy(normalize=False, nan_fill_value=0.0)
    #diagrams = vr.fit_transform(points[None, :, :])
    diagrams = wap.fit_transform(points[None, :, :])
    entropy = pe.fit_transform(diagrams)[0]   # [H0, H1, H2]
    return float(entropy[0]), float(entropy[1]), float(entropy[2])

def process_one(sample_id, data_root, class_to_idx, num_points, do_normalize, dataset_name):
    class_name = parse_sample_id(sample_id, dataset_name)
    class_idx = class_to_idx[class_name]
    txt_name = ensure_txt_filename(sample_id)
    txt_path = data_root / class_name / txt_name
    points = load_point_cloud(txt_path)
    points = fix_num_points(points, num_points=num_points)

    if do_normalize:
        points = normalize_points(points)
    h0, h1, h2 = compute_entropy(points)

    return {
        "sample_id": sample_id,
        "class_name": class_name,
        "class_idx": class_idx,
        "H0": h0,
        "H1": h1,
        "H2": h2
    }

def compute_split(split_ids, data_root, class_to_idx, num_points, do_normalize, n_jobs, dataset_name):
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(process_one)(sid, data_root, class_to_idx, num_points, do_normalize, dataset_name)
        for sid in tqdm(split_ids)
    )
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="modelnet40", choices=["modelnet40", "sonn"])
    parser.add_argument("--data_root", type=str, default="")
    parser.add_argument("--train_split", type=str, default="")
    parser.add_argument("--test_split", type=str, default="")
    parser.add_argument("--out_dir", type=str, default="")
    parser.add_argument("--num_points", type=int, default=2048)
    parser.add_argument("--n_jobs", type=int, default=14)
    parser.add_argument("--normalize", action="store_true")
    args = parser.parse_args()

    cfg = {
        "modelnet40": {
            "data_root": r"D:\GRE\PPT-Net\data\ModelNet40",
            "train_split": r"D:\GRE\PPT-Net\data\modelnet40_train.txt",
            "test_split": r"D:\GRE\PPT-Net\data\modelnet40_test.txt",
            "prefix": "mn40",
        },
        "sonn": {
            "data_root": r"D:\GRE\PPT-Net\data\SONNDataSet\SONN",
            "train_split": r"D:\GRE\PPT-Net\data\SONNDataSet\sonn_train.txt",
            "test_split": r"D:\GRE\PPT-Net\data\SONNDataSet\sonn_test.txt",
            "prefix": "sonn",
        },
    }

    c = cfg[args.dataset]

    if args.data_root == "":
        args.data_root = c["data_root"]

    if args.train_split == "":
        args.train_split = c["train_split"]

    if args.test_split == "":
        args.test_split = c["test_split"]

    if args.out_dir == "":
        base_out = Path(r"D:\GRE\PPT-Net\tda_entropy_project\outputs\ground_truths")
        out_dir = base_out / args.dataset
    else:
        out_dir = Path(args.out_dir) / args.dataset

    out_dir.mkdir(parents=True, exist_ok=True)

    data_root = Path(args.data_root)
    train_split = Path(args.train_split)
    test_split = Path(args.test_split)

    class_names = sorted([p.name for p in data_root.iterdir() if p.is_dir()])
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    train_ids = load_split_ids(train_split)
    test_ids = load_split_ids(test_split)

    print(f"Dataset       : {args.dataset}")
    print(f"Train samples: {len(train_ids)}")
    print(f"Test samples : {len(test_ids)}")
    print(f"\nUsing {args.n_jobs} parallel workers (joblib)")
    print("\nComputing TRAIN entropy...")
    train_df = compute_split(train_ids, data_root, class_to_idx, args.num_points, args.normalize, args.n_jobs, args.dataset)
    print(train_df.head())

    print("\nComputing TEST entropy...")
    test_df = compute_split(test_ids, data_root, class_to_idx, args.num_points, args.normalize, args.n_jobs, args.dataset)

    train_out = out_dir / f"{c['prefix']}_train_entropy.csv"
    test_out = out_dir / f"{c['prefix']}_test_entropy.csv"

    train_df.to_csv(train_out, index=False)
    test_df.to_csv(test_out, index=False)

    print("\nDone.")
    print(f"Saved: {train_out}")
    print(f"Saved: {test_out}")


if __name__ == "__main__":
    main()