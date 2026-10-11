"""
make_figures_v2.py

Publication-clean plotting script for the corrected HEA benchmark.

This version intentionally DOES NOT regenerate the numerical parity plot
(Figure 4 in the earlier revision), because the current numerical benchmark
mixes arithmetic ability with recall of tabulated materials data.

Instead, it keeps only the defensible outputs from the current rerun and
writes every panel as a separate image:

    figure_phase_confusion_matrix.png
    figure_phase_performance.png
    figure_reproducibility_dH_sd.png
    figure_phase_agreement.png

Run after:
    python score_predictions.py

The script reads CSV files from config.DATA_DIR (normally data_rerun/).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import DATA_DIR


TIER_ORDER = [
    "Simple (Binary)",
    "Complex (Ternary)",
    "Very Complex (Quinary)",
]

TIER_SHORT = {
    "Simple (Binary)": "Binary",
    "Complex (Ternary)": "Ternary",
    "Very Complex (Quinary)": "Quinary",
}


# ---------------------------------------------------------------------
# General plotting settings
# ---------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.linewidth": 1.0,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def need(name):
    path = DATA_DIR / name
    if not path.exists():
        raise SystemExit(
            f"Missing required file: {path}\n"
            "Run score_predictions.py first."
        )
    return path


def save(fig, filename):
    out = DATA_DIR / filename
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


# =====================================================================
# FIGURE A — Phase-classification confusion matrix
# =====================================================================
def plot_phase_confusion_matrix():
    cm = pd.read_csv(need("confusion_matrix.csv"), index_col=0)

    fig, ax = plt.subplots(figsize=(6.8, 5.6))

    vmax = max(1, int(np.nanmax(cm.values)))
    image = ax.imshow(cm.values, cmap="Blues", vmin=0, vmax=vmax)

    ax.set_xticks(np.arange(len(cm.columns)))
    ax.set_yticks(np.arange(len(cm.index)))
    ax.set_xticklabels(cm.columns)
    ax.set_yticklabels(cm.index)

    ax.set_xlabel("GPT-predicted phase")
    ax.set_ylabel("Reference phase")
    ax.set_title("Phase-classification confusion matrix", fontweight="bold")

    threshold = vmax / 2.0

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = int(cm.iloc[i, j])
            ax.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                fontweight="bold",
                fontsize=13,
                color="white" if value > threshold else "black",
            )

    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Count")

    save(fig, "figure_phase_confusion_matrix.png")


# =====================================================================
# FIGURE B — Accuracy and macro-F1 by complexity
# =====================================================================
def plot_phase_performance():
    metrics = pd.read_csv(need("metrics_classification.csv"))

    tier = (
        metrics[metrics["tier"] != "Overall"]
        .set_index("tier")
        .reindex(TIER_ORDER)
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    x = np.arange(len(tier))
    width = 0.36

    bars_accuracy = ax.bar(
        x - width / 2,
        tier["accuracy"],
        width,
        label="Accuracy",
    )

    bars_f1 = ax.bar(
        x + width / 2,
        tier["macro_F1"],
        width,
        label="Macro-F1",
    )

    for bar, value in zip(bars_accuracy, tier["accuracy"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.025,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    for bar, value in zip(bars_f1, tier["macro_F1"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.025,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([TIER_SHORT[t] for t in TIER_ORDER])
    ax.set_ylim(0.0, 1.10)
    ax.set_ylabel("Score")
    ax.set_title(
        "Phase-classification performance by complexity",
        fontweight="bold",
    )
    ax.legend(frameon=True)
    ax.grid(axis="y", alpha=0.22)

    save(fig, "figure_phase_performance.png")


# =====================================================================
# FIGURE C — Numerical reproducibility only
# Mean within-system run-to-run SD of ΔHmix.
# This replaces the unstable coefficient-of-variation metric.
# =====================================================================
def plot_dh_reproducibility():
    rp = (
        pd.read_csv(need("metrics_reproducibility.csv"))
        .set_index("tier")
        .reindex(TIER_ORDER)
        .reset_index()
    )

    required = "mean_SD_dH_kJmol"
    if required not in rp.columns:
        raise SystemExit(
            f"{required!r} not found in metrics_reproducibility.csv.\n"
            "Use the corrected score_predictions.py before running this script."
        )

    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    x = np.arange(len(rp))
    bars = ax.bar(x, rp[required], width=0.58)

    for bar, value in zip(bars, rp[required]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(0.02, 0.02 * rp[required].max()),
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([TIER_SHORT[t] for t in TIER_ORDER])
    ax.set_ylabel(
        r"Mean run-to-run SD of $\Delta H_{mix}$ (kJ mol$^{-1}$)"
    )
    ax.set_title(
        "Numerical reproducibility across five stochastic runs",
        fontweight="bold",
    )
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.22)

    save(fig, "figure_reproducibility_dH_sd.png")


# =====================================================================
# FIGURE D — Categorical reproducibility only
# Kept separate from numerical SD to avoid a dual-y-axis figure.
# =====================================================================
def plot_phase_agreement():
    rp = (
        pd.read_csv(need("metrics_reproducibility.csv"))
        .set_index("tier")
        .reindex(TIER_ORDER)
        .reset_index()
    )

    required = "categorical_agreement_pct"
    if required not in rp.columns:
        raise SystemExit(
            f"{required!r} not found in metrics_reproducibility.csv.\n"
            "Use the corrected score_predictions.py before running this script."
        )

    fig, ax = plt.subplots(figsize=(7.2, 5.0))

    x = np.arange(len(rp))
    y = rp[required].astype(float).values

    ax.plot(x, y, marker="o", linewidth=2)

    for xx, value in zip(x, y):
        ax.text(
            xx,
            min(value + 1.5, 103.0),
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([TIER_SHORT[t] for t in TIER_ORDER])
    ax.set_ylim(0, 105)
    ax.set_ylabel("Majority agreement across five runs (%)")
    ax.set_title(
        "Phase-prediction reproducibility",
        fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.22)

    save(fig, "figure_phase_agreement.png")


def main():
    print(f"Reading scored results from: {DATA_DIR}")
    print("Numerical parity plot intentionally omitted in make_figures_v2.py.")

    plot_phase_confusion_matrix()
    plot_phase_performance()
    plot_dh_reproducibility()
    plot_phase_agreement()

    print("\nDone.")
    print("Generated four separate publication figures:")
    print("  figure_phase_confusion_matrix.png")
    print("  figure_phase_performance.png")
    print("  figure_reproducibility_dH_sd.png")
    print("  figure_phase_agreement.png")


if __name__ == "__main__":
    main()
