import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'text.usetex': True})

output_dir = Path("plots")
output_dir.mkdir(exist_ok=True)
# ============================================================
# Paths
# ============================================================
dataset = "sonn"   # "modelnet40" or "sonn"
#types = ["amplitude_betti", "amplitude_bottleneck","amplitude_heat","amplitude_landscape","amplitude_persistence_image","amplitude_silhouette","amplitude_wasserstein"]
types = ["entropy"]
for t in types:
    if t != "entropy": # for amplitude types
        if dataset == "modelnet40":
            pptnet_json = Path(
                rf"D:\Code\1D-TDA-Vector-Prediction\Data\OneDrive_1_5-11-2026\Ours amplitudes MN40\{t}\{t}\train_results\best_metrics.json")
            ripsnet_json = Path(
                rf"D:\Code\1D-TDA-Vector-Prediction\Data\NEW ripsnet_50ep_results\ripsnet_50ep_results\amplitudes\{t}\ripsnet_out_50ep\best_test_metrics.json")
            dataset_title = "ModelNet40"
        else:
            pptnet_json = Path(
                rf"D:\Code\1D-TDA-Vector-Prediction\Data\OneDrive_1_5-11-2026\Ours Amplitudes SONN\SONN_amplitude_outputs\{t}\{t}\train_results\best_metrics.json")
            ripsnet_json = Path(
                rf"D:\Code\1D-TDA-Vector-Prediction\Data\NEW ripsnet_50ep_results\ripsnet_50ep_results\SONN_amplitudes\{t}\ripsnet_out_50ep\best_test_metrics.json")
            dataset_title = "SONN"
    if t == "entropy":
        if dataset == "modelnet40":
            pptnet_json = Path(rf"D:\Code\1D-TDA-Vector-Prediction\Data\OneDrive_1_5-11-2026\Ours entropy MN40\train_entropy_50\best_test_metrics.json")
            ripsnet_json = Path(rf"D:\Code\1D-TDA-Vector-Prediction\Data\NEW ripsnet_50ep_results\mn40_ripsnet_entropy_50ep_results\ripsnet_entropy_50ep_out\best_test_metrics.json")
            dataset_title = "ModelNet40"
        else:
            pptnet_json = Path(rf"D:\Code\1D-TDA-Vector-Prediction\Data\OneDrive_1_5-11-2026\Ours entropy SONN\SONN_entropy_outputs\sonn_train_results\sonn\best_test_metrics.json")
            ripsnet_json = Path(rf"D:\Code\1D-TDA-Vector-Prediction\Data\NEW ripsnet_50ep_results\ripsnet_50ep_results\SONN_entropy\ripsnet_out_50ep\best_test_metrics.json")
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
    models = ["Ours", "RipsNet"]

    # ============================================================
    # Metrics to compare
    # ============================================================
    overall_metrics = ["mse", "mae", "rmse", "r2_mean"]
    per_dim_metrics = [
        "mae_H0", "mae_H1", "mae_H2",
        "rmse_H0", "rmse_H1", "rmse_H2",
        "r2_H0", "r2_H1", "r2_H2"
    ]
    width = 0.35

    # ============================================================
    # Helper function to extract values
    # ============================================================
    def get_vals(metric):
        return [pptnet.get(metric, np.nan), ripsnet.get(metric, np.nan)]

    # ============================================================
    # Plot 1: Overall metrics
    # ============================================================
    # x = np.arange(len(overall_metrics))
    # width = 0.35
    #
    # ppt_vals = [pptnet.get(m, np.nan) for m in overall_metrics]
    # rip_vals = [ripsnet.get(m, np.nan) for m in overall_metrics]
    #
    # plt.figure(figsize=(10, 6))
    # bars1 = plt.bar(x - width/2, ppt_vals, width, label="Ours")
    # bars2 = plt.bar(x + width/2, rip_vals, width, label="RipsNet")
    #
    # plt.xticks(x, overall_metrics, fontsize=11)
    # plt.ylabel("Value", fontsize=12)
    # plt.title(f"{dataset_title} Best Test Metrics: Ours vs RipsNet", fontsize=14)
    # plt.legend(fontsize=14)
    # plt.grid(axis="y", linestyle="--", alpha=0.4)
    #
    # for bars in [bars1, bars2]:
    #     for b in bars:
    #         h = b.get_height()
    #         plt.text(
    #             b.get_x() + b.get_width()/2,
    #             h,
    #             f"{h:.3f}",
    #             ha="center",
    #             va="bottom",
    #             fontsize=9
    #         )
    #
    # plt.tight_layout()
    # # plt.savefig(output_dir /f"{t}_overall_metrics_{dataset}.png", dpi=300)
    # plt.show()

    # ============================================================
    # Plot 2: MAE per homology dimension
    # ============================================================
    mae_metrics = ["mae_H0", "mae_H1", "mae_H2"]
    x = np.arange(len(mae_metrics))

    ppt_vals = [pptnet.get(m, np.nan) for m in mae_metrics]
    rip_vals = [ripsnet.get(m, np.nan) for m in mae_metrics]

    plt.figure(figsize=(8, 5))
    bars1 = plt.bar(x - width/2, ppt_vals, width, label="Ours")
    bars2 = plt.bar(x + width/2, rip_vals, width, label="RipsNet")

    plt.xticks(x, ["$H_0$", "$H_1$", "$H_2$"], fontsize=18)
    plt.ylabel("MAE", fontsize=18)
    plt.title("MAE by Homology Dimension", fontsize=14)
    plt.legend(fontsize=14)
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    # for bars in [bars1, bars2]:
    #     for b in bars:
    #         h = b.get_height()
    #         plt.text(
    #             b.get_x() + b.get_width()/2,
    #             h,
    #             f"{h:.3f}",
    #             ha="center",
    #             va="bottom",
    #             fontsize=9
    #         )

    plt.tight_layout()
    plt.savefig(output_dir / f"{t}_mae_comparison_{dataset}.png", dpi=300)
    plt.show()

    # ============================================================
    # Plot 3: MSE per homology dimension
    # ============================================================
    rmse_metrics = ["rmse_H0", "rmse_H1", "rmse_H2"]
    x = np.arange(len(rmse_metrics))

    ppt_vals = np.square([pptnet.get(m, np.nan) for m in rmse_metrics])
    rip_vals = np.square([ripsnet.get(m, np.nan) for m in rmse_metrics])


    plt.figure(figsize=(8, 5))
    bars1 = plt.bar(x - width/2, ppt_vals, width, label="Ours")
    bars2 = plt.bar(x + width/2, rip_vals, width, label="RipsNet")

    plt.xticks(x, ["$H_0$", "$H_1$", "$H_2$"], fontsize=18)
    plt.ylabel("MSE", fontsize=18)
    plt.title("MSE by Homology Dimension", fontsize=14)
    plt.legend(fontsize=14)
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    # for bars in [bars1, bars2]:
    #     for b in bars:
    #         h = b.get_height()
    #         plt.text(
    #             b.get_x() + b.get_width()/2,
    #             h,
    #             f"{h:.3f}",
    #             ha="center",
    #             va="bottom",
    #             fontsize=9
    #         )

    plt.tight_layout()
    plt.savefig(output_dir /f"{t}_mse_comparison_{dataset}.png", dpi=300)
    plt.show()

    # ============================================================
    # Plot 4: R² per homology dimension
    # ============================================================
    r2_metrics = ["r2_H0", "r2_H1", "r2_H2"]
    x = np.arange(len(r2_metrics))

    ppt_vals = [pptnet.get(m, np.nan) for m in r2_metrics]
    rip_vals = [ripsnet.get(m, np.nan) for m in r2_metrics]

    plt.figure(figsize=(8, 5))
    bars1 = plt.bar(x - width/2, ppt_vals, width, label="Ours")
    bars2 = plt.bar(x + width/2, rip_vals, width, label="RipsNet")

    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xticks(x, ["$H_0$", "$H_1$", "$H_2$"], fontsize=18)
    plt.ylabel("R$^2$", fontsize=18)
    plt.title("R$^2$ by Homology Dimension", fontsize=14)
    plt.legend(fontsize=14)
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    # for bars in [bars1, bars2]:
    #     for b in bars:
    #         h = b.get_height()
    #         offset = 0.5 if h >= 0 else -1.5
    #         plt.text(
    #             b.get_x() + b.get_width()/2,
    #             h + offset,
    #             f"{h:.3f}",
    #             ha="center",
    #             va="bottom" if h >= 0 else "top",
    #             fontsize=9
    #         )

    plt.tight_layout()
    plt.savefig(output_dir /f"{t}_r2_comparison_{dataset}.png", dpi=300)
    plt.show()

    print("Saving plots to:", output_dir.resolve())

    # ============================================================
    # Print numeric summary
    # ============================================================
    # print(f"\n================ BEST TEST METRICS SUMMARY for {t} ================\n")
    # print("Ours:")
    # for k, v in pptnet.items():
    #     print(f"  {k:10s}: {v}")
    #
    # print("\n RipsNet:")
    # for k, v in ripsnet.items():
    #     print(f"  {k:10s}: {v}")
