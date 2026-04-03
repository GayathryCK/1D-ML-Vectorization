import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import argparse

# ============================================================
# Choose dataset and metric here
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", choices=["modelnet40", "sonn"], required=True)
parser.add_argument("--metric", required=True)
args = parser.parse_args()

dataset = args.dataset
metric = args.metric

# ============================================================
# Resolve paths
# ============================================================
if dataset == "modelnet40":
    pptnet_json = Path(
        rf"D:\GRE\PPT-Net\tda_vector_project\outputs\amplitude_{metric}\train_results\best_metrics.json"
    )
    ripsnet_json = Path(
        rf"D:\GRE\PPT-Net\From_Ram\ripsnet_mn40_results\ripsnet_amplitude_{metric}_results\best_test_metrics.json"
    )
    output_dir = Path(
        rf"D:\GRE\PPT-Net\tda_vector_project\plots\ModelNet\{metric}"
    )
    dataset_title = "ModelNet40"
elif dataset == "sonn":
    pptnet_json = Path(
        rf"D:\GRE\PPT-Net\tda_vector_project\SONN_outputs\amplitude_{metric}\train_results\best_metrics.json"
    )
    ripsnet_json = Path(
        rf"D:\GRE\PPT-Net\From_Ram\ripsnet_SONN_results\ripsnet_sonn_amplitude_{metric}_results\best_test_metrics.json"
    )
    output_dir = Path(
        rf"D:\GRE\PPT-Net\tda_vector_project\plots\SONN\{metric}"
    )
    dataset_title = "SONN"
else:
    raise ValueError(f"Unsupported dataset: {dataset}")

output_dir.mkdir(parents=True, exist_ok=True)

# ============================================================
# Load JSON files
# ============================================================
with open(pptnet_json, "r", encoding="utf-8") as f:
    pptnet = json.load(f)

with open(ripsnet_json, "r", encoding="utf-8") as f:
    ripsnet = json.load(f)

# ============================================================
# Metrics
# ============================================================
overall_metrics = ["mse", "mae", "rmse", "r2_mean"]
mae_metrics = ["mae_H0", "mae_H1", "mae_H2"]
rmse_metrics = ["rmse_H0", "rmse_H1", "rmse_H2"]
r2_metrics = ["r2_H0", "r2_H1", "r2_H2"]

width = 0.35

# ============================================================
# Plot 1: Overall metrics
# ============================================================
x = np.arange(len(overall_metrics))
ppt_vals = [pptnet.get(m, np.nan) for m in overall_metrics]
rip_vals = [ripsnet.get(m, np.nan) for m in overall_metrics]

plt.figure(figsize=(10, 6))
bars1 = plt.bar(x - width/2, ppt_vals, width, label="PPT-Net")
bars2 = plt.bar(x + width/2, rip_vals, width, label="RipsNet")

plt.xticks(x, overall_metrics, fontsize=11)
plt.ylabel("Value", fontsize=12)
plt.title(f"{dataset_title} - {metric} - Overall Test Metrics", fontsize=14)
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.4)

for bars in [bars1, bars2]:
    for b in bars:
        h = b.get_height()
        plt.text(
            b.get_x() + b.get_width()/2,
            h,
            f"{h:.3f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

plt.tight_layout()
plt.savefig(output_dir / "overall_metrics.png", dpi=300)
plt.close()

# ============================================================
# Plot 2: MAE
# ============================================================
x = np.arange(len(mae_metrics))
ppt_vals = [pptnet.get(m, np.nan) for m in mae_metrics]
rip_vals = [ripsnet.get(m, np.nan) for m in mae_metrics]

plt.figure(figsize=(8, 5))
bars1 = plt.bar(x - width/2, ppt_vals, width, label="PPT-Net")
bars2 = plt.bar(x + width/2, rip_vals, width, label="RipsNet")

plt.xticks(x, ["H0", "H1", "H2"], fontsize=11)
plt.ylabel("MAE", fontsize=12)
plt.title(f"{dataset_title} - {metric} - MAE", fontsize=14)
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.4)

for bars in [bars1, bars2]:
    for b in bars:
        h = b.get_height()
        plt.text(
            b.get_x() + b.get_width()/2,
            h,
            f"{h:.3f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

plt.tight_layout()
plt.savefig(output_dir / "mae_comparison.png", dpi=300)
plt.close()

# ============================================================
# Plot 3: RMSE
# ============================================================
x = np.arange(len(rmse_metrics))
ppt_vals = [pptnet.get(m, np.nan) for m in rmse_metrics]
rip_vals = [ripsnet.get(m, np.nan) for m in rmse_metrics]

plt.figure(figsize=(8, 5))
bars1 = plt.bar(x - width/2, ppt_vals, width, label="PPT-Net")
bars2 = plt.bar(x + width/2, rip_vals, width, label="RipsNet")

plt.xticks(x, ["H0", "H1", "H2"], fontsize=11)
plt.ylabel("RMSE", fontsize=12)
plt.title(f"{dataset_title} - {metric} - RMSE", fontsize=14)
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.4)

for bars in [bars1, bars2]:
    for b in bars:
        h = b.get_height()
        plt.text(
            b.get_x() + b.get_width()/2,
            h,
            f"{h:.3f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

plt.tight_layout()
plt.savefig(output_dir / "rmse_comparison.png", dpi=300)
plt.close()

# ============================================================
# Plot 4: R2
# ============================================================
x = np.arange(len(r2_metrics))
ppt_vals = [pptnet.get(m, np.nan) for m in r2_metrics]
rip_vals = [ripsnet.get(m, np.nan) for m in r2_metrics]

plt.figure(figsize=(8, 5))
bars1 = plt.bar(x - width/2, ppt_vals, width, label="PPT-Net")
bars2 = plt.bar(x + width/2, rip_vals, width, label="RipsNet")

plt.axhline(0, linestyle="--", linewidth=1)
plt.xticks(x, ["H0", "H1", "H2"], fontsize=11)
plt.ylabel("R²", fontsize=12)
plt.title(f"{dataset_title} - {metric} - R²", fontsize=14)
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.4)

for bars in [bars1, bars2]:
    for b in bars:
        h = b.get_height()
        offset = max(0.01, abs(h) * 0.03)
        y = h + offset if h >= 0 else h - offset
        plt.text(
            b.get_x() + b.get_width()/2,
            y,
            f"{h:.3f}",
            ha="center",
            va="bottom" if h >= 0 else "top",
            fontsize=9
        )

plt.tight_layout()
plt.savefig(output_dir / "r2_comparison.png", dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved plots to: {output_dir}")