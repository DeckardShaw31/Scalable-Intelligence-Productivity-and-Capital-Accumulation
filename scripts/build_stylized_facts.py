"""Plot source-reported AI scaling facts used to discipline scenario ranges.

The input is a small, auditable table of headline trend estimates reported by Epoch
AI.  It is not a reconstruction of the underlying model-level database and is not
used to estimate structural parameters.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "source_reported_ai_scaling.csv"
FIGURE_DIR = ROOT / "figures"


def main():
    data = pd.read_csv(DATA)
    required = {"metric", "annual_multiplier", "doubling_months", "period", "source"}
    if set(data.columns) != required:
        raise ValueError("Unexpected columns in the source-reported facts table")
    if (data["annual_multiplier"] <= 1.0).any():
        raise ValueError("Annual multipliers must exceed one")

    implied_months = 12.0 * np.log(2.0) / np.log(data["annual_multiplier"])
    if np.max(np.abs(implied_months - data["doubling_months"])) > 0.15:
        raise ValueError("Multiplier and doubling-time entries are inconsistent")

    order = np.arange(len(data))[::-1]
    colors = ["#2f5597", "#d97904", "#4c956c", "#8a5a9e"]
    fig, ax = plt.subplots(figsize=(8.6, 3.8), constrained_layout=True)
    bars = ax.barh(
        order,
        data["annual_multiplier"],
        color=colors,
        edgecolor="white",
        height=0.68,
    )
    ax.set_yticks(order, data["metric"])
    ax.set_xlabel("Reported annual multiplication factor")
    ax.set_xlim(0.0, 5.6)
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)

    for bar, multiplier, months in zip(
        bars, data["annual_multiplier"], data["doubling_months"]
    ):
        ax.text(
            multiplier + 0.08,
            bar.get_y() + bar.get_height() / 2.0,
            f"{multiplier:.1f}×/yr  ({months:.1f} mo.)",
            va="center",
            fontsize=9,
        )

    ax.set_title("Selected source-reported AI scaling trends")
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / "ai-scaling-facts.pdf", bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "ai-scaling-facts.png", dpi=220, bbox_inches="tight")
    print(data.to_string(index=False))


if __name__ == "__main__":
    main()