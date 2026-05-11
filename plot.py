import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

output_dir = Path("plots")
output_dir.mkdir(exist_ok=True)
# ============================================================
# Paths
# ============================================================
dataset = "modelnet40"   # "modelnet40" or "sonn"

if dataset == "modelnet40":
    pptnet_json = Path(r"D:\Code\1D-TDA-Vector-Prediction\Data\OneDrive_1_5-11-2026\Ours amplitudes MN40\amplitude_betti\amplitude_betti\train_results\best_metrics.json")
    ripsnet_json = Path(r"D:\Code\1D-TDA-Vector-Prediction\Data\OneDrive_1_5-11-2026\RIPSNET amplitudes MN40\ripsnet_mn40_results\ripsnet_amplitude_betti_results\best_test_metrics.json")
    dataset_title = "ModelNet40"
else:
    pptnet_json = Path(r"D:\GRE\PPT-Net\tda_entropy_project\outputs\sonn\train_entropy\best_test_metrics.json")
    ripsnet_json = Path(r"D:\GRE\PPT-Net\tda_entropy_project\outputs\sonn\ripsnet_results\best_test_metrics.json")
    dataset_title = "SONN"

# ============================================================
# Load JSON files
# ============================================================
with open(pptnet_json, "r") as f:
    pptnet = json.load(f)

with open(ripsnet_json, "r") as f:
    ripsnet = json.load(f)

# ============================================================
# Model labels
# ============================================================
models = ["PPT-Net", "RipsNet"]

# ============================================================
# Metrics to compare
# ============================================================
overall_metrics = ["mse", "mae", "rmse", "r2_mean"]
per_dim_metrics = [
    "mae_H0", "mae_H1", "mae_H2",
    "rmse_H0", "rmse_H1", "rmse_H2",
    "r2_H0", "r2_H1", "r2_H2"
]

# ============================================================
# Helper function to extract values
# ============================================================
def get_vals(metric):
    return [pptnet.get(metric, np.nan), ripsnet.get(metric, np.nan)]

# ============================================================
# Plot 1: Overall metrics
# ============================================================
x = np.arange(len(overall_metrics))
width = 0.35

ppt_vals = [pptnet.get(m, np.nan) for m in overall_metrics]
rip_vals = [ripsnet.get(m, np.nan) for m in overall_metrics]

plt.figure(figsize=(10, 6))
bars1 = plt.bar(x - width/2, ppt_vals, width, label="PPT-Net")
bars2 = plt.bar(x + width/2, rip_vals, width, label="RipsNet")

plt.xticks(x, overall_metrics, fontsize=11)
plt.ylabel("Value", fontsize=12)
plt.title(f"{dataset_title} Best Test Metrics: PPT-Net vs RipsNet", fontsize=14)
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
plt.savefig(output_dir /"overall_metrics.png", dpi=300)
plt.show()

# ============================================================
# Plot 2: MAE per homology dimension
# ============================================================
mae_metrics = ["mae_H0", "mae_H1", "mae_H2"]
x = np.arange(len(mae_metrics))

ppt_vals = [pptnet.get(m, np.nan) for m in mae_metrics]
rip_vals = [ripsnet.get(m, np.nan) for m in mae_metrics]

plt.figure(figsize=(8, 5))
bars1 = plt.bar(x - width/2, ppt_vals, width, label="PPT-Net")
bars2 = plt.bar(x + width/2, rip_vals, width, label="RipsNet")

plt.xticks(x, ["H0", "H1", "H2"], fontsize=11)
plt.ylabel("MAE", fontsize=12)
plt.title("MAE by Homology Dimension", fontsize=14)
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
plt.savefig(output_dir /"mae_comparison.png", dpi=300)
plt.show()

# ============================================================
# Plot 3: RMSE per homology dimension
# ============================================================
rmse_metrics = ["rmse_H0", "rmse_H1", "rmse_H2"]
x = np.arange(len(rmse_metrics))

ppt_vals = [pptnet.get(m, np.nan) for m in rmse_metrics]
rip_vals = [ripsnet.get(m, np.nan) for m in rmse_metrics]

plt.figure(figsize=(8, 5))
bars1 = plt.bar(x - width/2, ppt_vals, width, label="PPT-Net")
bars2 = plt.bar(x + width/2, rip_vals, width, label="RipsNet")

plt.xticks(x, ["H0", "H1", "H2"], fontsize=11)
plt.ylabel("RMSE", fontsize=12)
plt.title("RMSE by Homology Dimension", fontsize=14)
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
plt.savefig(output_dir /"rmse_comparison.png", dpi=300)
plt.show()

# ============================================================
# Plot 4: R² per homology dimension
# ============================================================
r2_metrics = ["r2_H0", "r2_H1", "r2_H2"]
x = np.arange(len(r2_metrics))

ppt_vals = [pptnet.get(m, np.nan) for m in r2_metrics]
rip_vals = [ripsnet.get(m, np.nan) for m in r2_metrics]

plt.figure(figsize=(8, 5))
bars1 = plt.bar(x - width/2, ppt_vals, width, label="PPT-Net")
bars2 = plt.bar(x + width/2, rip_vals, width, label="RipsNet")

plt.axhline(0, linestyle="--", linewidth=1)
plt.xticks(x, ["H0", "H1", "H2"], fontsize=11)
plt.ylabel("R²", fontsize=12)
plt.title("R² by Homology Dimension", fontsize=14)
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.4)

for bars in [bars1, bars2]:
    for b in bars:
        h = b.get_height()
        offset = 0.5 if h >= 0 else -1.5
        plt.text(
            b.get_x() + b.get_width()/2,
            h + offset,
            f"{h:.3f}",
            ha="center",
            va="bottom" if h >= 0 else "top",
            fontsize=9
        )

plt.tight_layout()
plt.savefig(output_dir /"r2_comparison.png", dpi=300)
plt.show()

print("Saving plots to:", output_dir.resolve())

# ============================================================
# Print numeric summary
# ============================================================
print("\n================ BEST TEST METRICS SUMMARY ================\n")
print("PPT-Net:")
for k, v in pptnet.items():
    print(f"  {k:10s}: {v}")

print("\nRipsNet:")
for k, v in ripsnet.items():
    print(f"  {k:10s}: {v}")