import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from gtda.homology import WeakAlphaPersistence
from gtda.diagrams import Amplitude

from vector_utils import (
    load_split_ids,
    load_point_cloud,
    normalize_points,
    sample_points,
    stable_seed_from_string,
)

VALID_AMPLITUDE_METRICS = [
    "bottleneck",
    "wasserstein",
    "betti",
    "landscape",
    "silhouette",
    "heat",
    "persistence_image",
]

DATASET_CONFIG = {
    "modelnet40": {
        "data_root": r"D:\GRE\PPT-Net\data\ModelNet40",
        "train_split": r"D:\GRE\PPT-Net\data\modelnet40_train.txt",
        "test_split": r"D:\GRE\PPT-Net\data\modelnet40_test.txt",
        "out_root": r"D:\GRE\PPT-Net\tda_vector_project\outputs",
        "prefix": "mn40",
        "title": "ModelNet40",
    },
    "sonn": {
        "data_root": r"D:\GRE\PPT-Net\data\SONNDataSet\SONN",
        "train_split": r"D:\GRE\PPT-Net\data\SONNDataSet\SONN_train.txt",
        "test_split": r"D:\GRE\PPT-Net\data\SONNDataSet\SONN_test.txt",
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


def compute_amplitude_vector(points: np.ndarray, metric: str) -> np.ndarray:
    points = np.unique(points, axis=0)

    wap = WeakAlphaPersistence(homology_dimensions=[0, 1, 2], n_jobs=1)
    diagrams = wap.fit_transform(points[None, :, :])

    dgm = diagrams[0]

    finite_mask = np.isfinite(dgm[:, 0]) & np.isfinite(dgm[:, 1]) & np.isfinite(dgm[:, 2])
    dgm = dgm[finite_mask]

    if dgm.shape[0] == 0:
        return np.zeros(3, dtype=np.float32)

    diagrams_clean = dgm[None, :, :]

    amp = Amplitude(metric=metric, order=None, n_jobs=1)
    vec = amp.fit_transform(diagrams_clean)[0]

    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)

    if len(vec) != 3:
        raise ValueError(f"Expected amplitude vector length 3, got {len(vec)}: {vec}")

    return vec


def process_one(sample_id, data_root, class_to_idx, num_points, do_normalize, base_seed, amplitude_metric, dataset_name):
    class_name = parse_sample_id(sample_id, dataset_name)
    class_idx = class_to_idx[class_name]

    txt_name = ensure_txt_filename(sample_id)
    txt_path = data_root / class_name / txt_name
    points = load_point_cloud(txt_path)

    seed = stable_seed_from_string(sample_id, base_seed)
    points = sample_points(points, num_points=num_points, seed=seed)

    if do_normalize:
        points = normalize_points(points)

    vec = compute_amplitude_vector(points, metric=amplitude_metric)

    row = {
        "sample_id": sample_id,
        "class_name": class_name,
        "class_idx": class_idx,
        "H0": float(vec[0]),
        "H1": float(vec[1]),
        "H2": float(vec[2]),
    }
    return row


def compute_split(split_ids, data_root, class_to_idx, num_points, do_normalize, n_jobs, base_seed, amplitude_metric, dataset_name):
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(process_one)(
            sid,
            data_root,
            class_to_idx,
            num_points,
            do_normalize,
            base_seed,
            amplitude_metric,
            dataset_name,
        )
        for sid in tqdm(split_ids, desc=f"Computing {amplitude_metric}")
    )

    df = pd.DataFrame(results)
    ordered_cols = ["sample_id", "class_name", "class_idx", "H0", "H1", "H2"]
    df = df[ordered_cols]
    return df


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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amplitude_metric", type=str, required=True, choices=VALID_AMPLITUDE_METRICS)
    args = parser.parse_args()

    cfg = get_dataset_config(args.dataset)

    if args.data_root == "":
        args.data_root = cfg["data_root"]

    if args.train_split == "":
        args.train_split = cfg["train_split"]

    if args.test_split == "":
        args.test_split = cfg["test_split"]

    if args.out_dir == "":
        args.out_dir = str(Path(cfg["out_root"]) / f"amplitude_{args.amplitude_metric}" / "ground_truths")

    data_root = Path(args.data_root)
    train_split = Path(args.train_split)
    test_split = Path(args.test_split)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    class_names = sorted([p.name for p in data_root.iterdir() if p.is_dir()])
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    train_ids = load_split_ids(train_split)
    test_ids = load_split_ids(test_split)

    print("=" * 70)
    print(f"{cfg['title']} Amplitude Ground-Truth Generation")
    print("=" * 70)
    print(f"dataset           : {args.dataset}")
    print(f"data_root         : {data_root}")
    print(f"train_split       : {train_split}")
    print(f"test_split        : {test_split}")
    print(f"out_dir           : {out_dir}")
    print(f"num_points        : {args.num_points}")
    print(f"normalize         : {args.normalize}")
    print(f"n_jobs            : {args.n_jobs}")
    print(f"amplitude_metric  : {args.amplitude_metric}")
    print(f"seed              : {args.seed}")
    print(f"train samples     : {len(train_ids)}")
    print(f"test samples      : {len(test_ids)}")
    print("=" * 70)

    print("\nComputing TRAIN vectors...")
    train_df = compute_split(
        train_ids,
        data_root,
        class_to_idx,
        args.num_points,
        args.normalize,
        args.n_jobs,
        args.seed,
        args.amplitude_metric,
        args.dataset,
    )

    print("\nComputing TEST vectors...")
    test_df = compute_split(
        test_ids,
        data_root,
        class_to_idx,
        args.num_points,
        args.normalize,
        args.n_jobs,
        args.seed,
        args.amplitude_metric,
        args.dataset,
    )

    prefix = cfg["prefix"]
    train_out = out_dir / f"{prefix}_train_amplitude_{args.amplitude_metric}.csv"
    test_out = out_dir / f"{prefix}_test_amplitude_{args.amplitude_metric}.csv"

    train_df.to_csv(train_out, index=False)
    test_df.to_csv(test_out, index=False)

    print("\nDone.")
    print(f"Saved train CSV: {train_out}")
    print(f"Saved test  CSV: {test_out}")

    print("\nPreview:")
    print(train_df.head())
    print("\nTarget dimension detected: 3")


if __name__ == "__main__":
    main()