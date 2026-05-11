import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import pairwise_distances
from sklearn.manifold import TSNE

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import umap as umap_module

VALID_METRICS = [
    "bottleneck",
    "wasserstein",
    "betti",
    "landscape",
    "silhouette",
    "heat",
    "persistence_image",
]


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
def load_prediction_csv(pred_csv: Path):
    df = pd.read_csv(pred_csv)

    required_cols = [
        "sample_id", "class_name", "class_idx",
        "H0_true", "H1_true", "H2_true",
        "H0_pred", "H1_pred", "H2_pred",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {pred_csv}: {missing}")

    y_true = df[["H0_true", "H1_true", "H2_true"]].to_numpy(dtype=np.float32)
    y_pred = df[["H0_pred", "H1_pred", "H2_pred"]].to_numpy(dtype=np.float32)

    labels = df["class_name"].astype(str).to_numpy()
    sample_ids = df["sample_id"].astype(str).to_numpy()
    class_idxs = df["class_idx"].to_numpy()

    return df, y_true, y_pred, labels, sample_ids, class_idxs


# -----------------------------------------------------------------------------
# Metric 1: Distance matrix correlations
# -----------------------------------------------------------------------------
def compute_distance_matrix_correlations(y_true: np.ndarray, y_pred: np.ndarray):
    D_true = pairwise_distances(y_true, metric="euclidean")
    D_pred = pairwise_distances(y_pred, metric="euclidean")

    iu = np.triu_indices_from(D_true, k=1)
    dt_flat = D_true[iu]
    dp_flat = D_pred[iu]

    pearson_val, pearson_p = pearsonr(dt_flat, dp_flat)
    spearman_val, spearman_p = spearmanr(dt_flat, dp_flat)

    metrics = {
        "pearson_distance_corr": float(pearson_val),
        "pearson_distance_corr_pvalue": float(pearson_p),
        "spearman_distance_corr": float(spearman_val),
        "spearman_distance_corr_pvalue": float(spearman_p),
    }
    return metrics, D_true, D_pred


# -----------------------------------------------------------------------------
# Metric 2: Neighbour overlap @ k
# -----------------------------------------------------------------------------
def get_knn_indices(X: np.ndarray, k: int) -> np.ndarray:
    D = pairwise_distances(X, metric="euclidean")
    np.fill_diagonal(D, np.inf)
    return np.argsort(D, axis=1)[:, :k]


def compute_overlap_at_k(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_ids: np.ndarray,
    labels: np.ndarray,
    class_idxs: np.ndarray,
    ks=(1, 5, 10),
):
    n = y_true.shape[0]
    rows = []
    summary = {}

    for k in ks:
        eff_k = min(k, n - 1)
        true_nn = get_knn_indices(y_true, eff_k)
        pred_nn = get_knn_indices(y_pred, eff_k)

        overlaps = []
        for i in range(n):
            tset = set(true_nn[i].tolist())
            pset = set(pred_nn[i].tolist())
            overlap = len(tset.intersection(pset)) / float(eff_k)
            overlaps.append(overlap)

            rows.append({
                "sample_index": int(i),
                "sample_id": sample_ids[i],
                "class_name": labels[i],
                "class_idx": int(class_idxs[i]),
                "k": int(k),
                "effective_k": int(eff_k),
                "overlap": float(overlap),
            })

        summary[f"overlap_at_{k}"] = float(np.mean(overlaps))

    return summary, pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Visualization helpers
# -----------------------------------------------------------------------------
def encode_labels(labels):
    unique = sorted(set(labels.tolist()))
    label_to_int = {lab: i for i, lab in enumerate(unique)}
    ints = np.array([label_to_int[x] for x in labels], dtype=np.int32)
    return ints, unique


def get_class_colors(unique_labels):
    cmap = plt.colormaps["tab20"]
    n_cls = len(unique_labels)
    colors = [cmap(i / max(n_cls - 1, 1)) for i in range(n_cls)]
    return colors


# -----------------------------------------------------------------------------
# Metric 3: t-SNE (joint fit)
# -----------------------------------------------------------------------------
def compute_tsne_joint(y_true: np.ndarray, y_pred: np.ndarray, seed: int = 42):
    n = y_true.shape[0]
    combined = np.vstack([y_true, y_pred])

    perplexity = min(30, max(5, n // 10))

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    )
    emb = tsne.fit_transform(combined)
    return emb[:n], emb[n:]


# -----------------------------------------------------------------------------
# Metric 4: UMAP (joint fit)
# -----------------------------------------------------------------------------
def compute_umap_joint(y_true: np.ndarray, y_pred: np.ndarray, seed: int = 42):

    n = y_true.shape[0]
    combined = np.vstack([y_true, y_pred])

    reducer = umap_module.UMAP(
        n_components=2,
        n_neighbors=min(15, n - 1),
        min_dist=0.1,
        metric="euclidean",
        random_state=seed,
    )
    emb = reducer.fit_transform(combined)
    return emb[:n], emb[n:]


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def plot_single_embedding(emb, labels, title, out_path):
    label_ids, unique_labels = encode_labels(labels)
    colors = get_class_colors(unique_labels)

    fig, ax = plt.subplots(figsize=(9, 7))
    for i, lab in enumerate(unique_labels):
        mask = label_ids == i
        ax.scatter(
            emb[mask, 0],
            emb[mask, 1],
            s=12,
            alpha=0.75,
            color=colors[i],
            label=lab
        )

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    if len(unique_labels) <= 20:
        ax.legend(markerscale=2, fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_overlay(emb_true, emb_pred, labels, title, out_path):
    label_ids, unique_labels = encode_labels(labels)
    colors = get_class_colors(unique_labels)

    fig, ax = plt.subplots(figsize=(10, 7))

    for i, lab in enumerate(unique_labels):
        mask = label_ids == i
        c = colors[i]

        ax.scatter(
            emb_true[mask, 0],
            emb_true[mask, 1],
            s=18,
            alpha=0.6,
            color=c,
            marker="o"
        )
        ax.scatter(
            emb_pred[mask, 0],
            emb_pred[mask, 1],
            s=18,
            alpha=0.6,
            color=c,
            marker="x"
        )

    class_handles = [
        mpatches.Patch(color=colors[i], label=lab)
        for i, lab in enumerate(unique_labels)
    ]
    style_handles = [
        plt.Line2D([0], [0], marker="o", color="grey", linestyle="None",
                   markersize=7, label="True"),
        plt.Line2D([0], [0], marker="x", color="grey", linestyle="None",
                   markersize=7, markeredgewidth=1.5, label="Predicted"),
    ]

    ax.legend(
        handles=class_handles + style_handles,
        fontsize=8,
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_csv", type=str, default="")
    parser.add_argument("--out_dir", type=str, default="")
    parser.add_argument("--amplitude_metric", type=str, default="bottleneck", choices=VALID_METRICS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    metric_root = Path(r"D:\GRE\PPT-Net\tda_vector_project\outputs") / f"amplitude_{args.amplitude_metric}"

    if args.pred_csv == "":
        args.pred_csv = str(metric_root / "train_results" / "best_test_predictions.csv")

    if args.out_dir == "":
        args.out_dir = str(metric_root / "structure_eval")

    pred_csv = Path(args.pred_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df, y_true, y_pred, labels, sample_ids, class_idxs = load_prediction_csv(pred_csv)

    print("=" * 70)
    print("Vector Structure Evaluation")
    print("=" * 70)
    print(f"pred_csv          : {pred_csv}")
    print(f"out_dir           : {out_dir}")
    print(f"amplitude_metric  : {args.amplitude_metric}")
    print(f"num_samples       : {len(df)}")
    print("=" * 70)

    # Distance matrix correlations
    dist_metrics, D_true, D_pred = compute_distance_matrix_correlations(y_true, y_pred)

    # Overlap @ k
    overlap_metrics, overlap_df = compute_overlap_at_k(
        y_true, y_pred, sample_ids, labels, class_idxs, ks=(1, 5, 10)
    )

    # Save structure metrics
    structure_metrics = {
        "num_samples": int(len(df)),
        **dist_metrics,
        **overlap_metrics,
    }

    with open(out_dir / "structure_metrics.json", "w", encoding="utf-8") as f:
        json.dump(structure_metrics, f, indent=2)

    overlap_df.to_csv(out_dir / "knn_overlap_per_sample.csv", index=False)

    print("Saved:")
    print(out_dir / "structure_metrics.json")
    print(out_dir / "knn_overlap_per_sample.csv")

    # t-SNE
    print("\nComputing joint t-SNE...")
    tsne_true, tsne_pred = compute_tsne_joint(y_true, y_pred, seed=args.seed)

    plot_single_embedding(
        tsne_true, labels,
        f"t-SNE True ({args.amplitude_metric})",
        out_dir / "tsne_true.png"
    )
    plot_single_embedding(
        tsne_pred, labels,
        f"t-SNE Predicted ({args.amplitude_metric})",
        out_dir / "tsne_pred.png"
    )
    plot_overlay(
        tsne_true, tsne_pred, labels,
        f"t-SNE Overlay: True (o) vs Predicted (x) [{args.amplitude_metric}]",
        out_dir / "tsne_overlay.png"
    )

    print("Saved:")
    print(out_dir / "tsne_true.png")
    print(out_dir / "tsne_pred.png")
    print(out_dir / "tsne_overlay.png")


        print("\nComputing joint UMAP...")
        umap_true, umap_pred = compute_umap_joint(y_true, y_pred, seed=args.seed)

        plot_single_embedding(
            umap_true, labels,
            f"UMAP True ({args.amplitude_metric})",
            out_dir / "umap_true.png"
        )
        plot_single_embedding(
            umap_pred, labels,
            f"UMAP Predicted ({args.amplitude_metric})",
            out_dir / "umap_pred.png"
        )
        plot_overlay(
            umap_true, umap_pred, labels,
            f"UMAP Overlay: True (o) vs Predicted (x) [{args.amplitude_metric}]",
            out_dir / "umap_overlay.png"
        )

        print("Saved:")
        print(out_dir / "umap_true.png")
        print(out_dir / "umap_pred.png")
        print(out_dir / "umap_overlay.png")

    print("\nSummary:")
    for k, v in structure_metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()