"""Generate two-dimensional sensitivity maps for the benchmark model.

The maps vary the elasticity of substitution and the scale of AI research
productivity. They are robustness exercises, not estimates or forecasts.
"""

from pathlib import Path
import importlib.util
import sys

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
FIGURE_DIR = ROOT / "figures"
SCRIPTS_DIR = ROOT / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

try:
    import simulate_regimes as model
except ImportError:
    spec = importlib.util.spec_from_file_location(
        "simulate_regimes", SCRIPTS_DIR / "simulate_regimes.py"
    )
    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)


def main():
    sigma_grid = np.linspace(0.4, 3.0, 27)
    research_scale_grid = np.linspace(0.5, 2.5, 25)
    terminal_labor_share = np.empty((research_scale_grid.size, sigma_grid.size))
    terminal_output_growth = np.empty_like(terminal_labor_share)

    base = {
        "lam": 0.45,
        "phi": 0.50,
        "xi": 0.25,
        "zeta": 0.08,
        "q": 1.0,
    }

    for row, scale in enumerate(research_scale_grid):
        for col, sigma in enumerate(sigma_grid):
            overrides = base | {
                "sigma": float(sigma),
                "chi": model.COMMON["chi"] * float(scale),
            }
            result = model.simulate(overrides, horizon=60.0, points=241)
            terminal_labor_share[row, col] = result["labor_share"][-1]
            terminal_output_growth[row, col] = result["output_growth"][-1]

    extent = [
        sigma_grid[0],
        sigma_grid[-1],
        research_scale_grid[0],
        research_scale_grid[-1],
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7), constrained_layout=True)

    share_image = axes[0].imshow(
        terminal_labor_share,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0 - model.COMMON["alpha"],
    )
    axes[0].set_title("Labor share after 60 years")
    share_bar = fig.colorbar(share_image, ax=axes[0])
    share_bar.set_label("Labor share")

    growth_image = axes[1].imshow(
        100.0 * terminal_output_growth,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="magma",
    )
    axes[1].set_title("Output growth after 60 years")
    growth_bar = fig.colorbar(growth_image, ax=axes[1])
    growth_bar.set_label("Percent per year")

    for ax in axes:
        ax.axvline(1.0, color="white", ls="--", lw=1.2, alpha=0.9)
        ax.set_xlabel("Elasticity of substitution, $\\sigma$")
        ax.set_ylabel("Research-productivity multiplier")

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / "sensitivity-map.pdf", bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "sensitivity-map.png", dpi=220, bbox_inches="tight")

    print(
        f"labor-share range: {terminal_labor_share.min():.3f}--"
        f"{terminal_labor_share.max():.3f}"
    )
    print(
        f"terminal-growth range: {100 * terminal_output_growth.min():.2f}%--"
        f"{100 * terminal_output_growth.max():.2f}%"
    )


if __name__ == "__main__":
    main()