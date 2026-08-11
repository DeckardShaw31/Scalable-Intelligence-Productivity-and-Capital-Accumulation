"""Generate illustrative transition paths for the scalable-intelligence model.

The exercises are deliberately not calibrated forecasts.  They show how changing
the substitution elasticity and the strength of research feedback changes the
qualitative dynamics of the same benchmark system.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp


ROOT = Path(__file__).resolve().parent.parent
FIGURE_DIR = ROOT / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


COMMON = {
    "alpha": 0.32,
    "omega": 0.65,
    "n": 0.01,
    "s_k": 0.18,
    "s_x": 0.16,
    "delta_k": 0.05,
    "delta_x": 0.12,
    "chi": 0.012,
    "xi": 0.12,
    "ell_r": 0.08,
    "u_r": 0.12,
    "zeta": 0.08,
    "q": 1.0,
}


SCENARIOS = {
    "Bottleneck ($\\sigma=0.6$)": {
        "sigma": 0.6,
        "lam": 0.15,
        "phi": 0.30,
    },
    "Substitution ($\\sigma=2$)": {
        "sigma": 2.0,
        "lam": 0.15,
        "phi": 0.30,
    },
    "Strong feedback + congestion": {
        "sigma": 2.0,
        "lam": 0.70,
        "phi": 0.70,
        "xi": 0.50,
        "chi": 0.035,
        "zeta": 0.025,
        "q": 1.00,
    },
}


def levels_and_flows(t, log_state, par):
    """Return model levels and flows from logarithmic state variables."""
    k, x, a = np.exp(log_state)
    labor = np.exp(par["n"] * t)
    labor_y = (1.0 - par["ell_r"]) * labor
    labor_r = par["ell_r"] * labor
    x_y = (1.0 - par["u_r"]) * x
    x_r = par["u_r"] * x

    machine = a * x_y
    nu = (par["sigma"] - 1.0) / par["sigma"]
    if np.isclose(par["sigma"], 1.0):
        cognition = labor_y ** par["omega"] * machine ** (1.0 - par["omega"])
        human_cognition_share = par["omega"]
    else:
        cognition_terms = (
            par["omega"] * labor_y**nu
            + (1.0 - par["omega"]) * machine**nu
        )
        cognition = cognition_terms ** (1.0 / nu)
        human_cognition_share = par["omega"] * labor_y**nu / cognition_terms
    output = k ** par["alpha"] * cognition ** (1.0 - par["alpha"])
    research = (
        par["chi"]
        * a ** par["lam"]
        * (labor_r + par["xi"] * a * x_r) ** par["phi"]
        / (1.0 + par["zeta"] * a ** par["q"])
    )
    labor_share = (1.0 - par["alpha"]) * human_cognition_share
    return k, x, a, labor, output, research, labor_share


def dynamics(t, log_state, par):
    k, x, a, _, output, research, _ = levels_and_flows(t, log_state, par)
    return np.array(
        [
            par["s_k"] * output / k - par["delta_k"],
            par["s_x"] * output / x - par["delta_x"],
            research / a,
        ]
    )


def simulate(overrides, horizon=80.0, points=801, initial=(4.0, 1.0, 1.0)):
    par = COMMON | overrides
    validate_parameters(par)
    grid = np.linspace(0.0, horizon, points)
    initial = np.log(np.asarray(initial, dtype=float))
    solution = solve_ivp(
        lambda t, y: dynamics(t, y, par),
        (grid[0], grid[-1]),
        initial,
        t_eval=grid,
        rtol=1e-9,
        atol=1e-11,
    )
    if not solution.success:
        raise RuntimeError(solution.message)

    series = [levels_and_flows(t, y, par) for t, y in zip(grid, solution.y.T)]
    k, x, a, labor, output, research, labor_share = map(np.asarray, zip(*series))
    state_growth = np.asarray(
        [dynamics(t, y, par) for t, y in zip(grid, solution.y.T)]
    )
    g_k, g_x, g_a = state_growth.T

    # Exact log derivative of the production function.  Within the CES cognition
    # aggregate, labor and machine expenditure shares sum to one.  Since allocation
    # shares are constant in these exercises, g_{L_Y}=n and g_{X_Y}=g_X.
    human_cognition_share = labor_share / (1.0 - par["alpha"])
    machine_cognition_share = 1.0 - human_cognition_share
    cognition_growth = (
        human_cognition_share * par["n"]
        + machine_cognition_share * (g_a + g_x)
    )
    output_growth = par["alpha"] * g_k + (1.0 - par["alpha"]) * cognition_growth

    result = {
        "t": grid,
        "k": k,
        "x": x,
        "a": a,
        "labor": labor,
        "output": output,
        "research": research,
        "labor_share": labor_share,
        "output_growth": output_growth,
    }
    validate_result(result, par)
    return result


def validate_parameters(par):
    """Check restrictions imposed by the benchmark model."""
    if not 0.0 < par["alpha"] < 1.0:
        raise ValueError("alpha must lie between zero and one")
    if not 0.0 < par["omega"] < 1.0:
        raise ValueError("omega must lie between zero and one")
    if par["sigma"] <= 0.0:
        raise ValueError("sigma must be positive")
    if par["s_k"] < 0.0 or par["s_x"] < 0.0 or par["s_k"] + par["s_x"] >= 1.0:
        raise ValueError("Saving shares must be nonnegative and sum to less than one")
    if not 0.0 <= par["ell_r"] < 1.0 or not 0.0 <= par["u_r"] < 1.0:
        raise ValueError("Research shares must leave positive inputs for production")
    for key in ("delta_k", "delta_x", "zeta", "q"):
        if par[key] < 0.0:
            raise ValueError(f"{key} must be nonnegative")
    for key in ("chi", "xi", "phi"):
        if par[key] <= 0.0:
            raise ValueError(f"{key} must be positive")


def validate_result(result, par):
    """Fail loudly if a numerical path violates basic model restrictions."""
    positive = ("k", "x", "a", "labor", "output", "research")
    for key in positive:
        values = result[key]
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError(f"Invalid values in {key}")

    upper_share = 1.0 - par["alpha"]
    shares = result["labor_share"]
    if np.any(shares < 0.0) or np.any(shares > upper_share + 1e-12):
        raise ValueError("Labor share lies outside its theoretical bounds")
    if not np.all(np.isfinite(result["output_growth"])):
        raise ValueError("Non-finite output growth")


def main():
    # The first two scenarios isolate substitution: all research parameters match.
    for key in ("lam", "phi"):
        assert SCENARIOS["Bottleneck ($\\sigma=0.6$)"][key] == SCENARIOS[
            "Substitution ($\\sigma=2$)"
        ][key]

    colors = ["#2f5597", "#d97904", "#4c956c"]
    results = {name: simulate(par) for name, par in SCENARIOS.items()}

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.4), constrained_layout=True)
    panels = [
        ("output", "Output index", True),
        ("a", "AI capability index", True),
        ("output_growth", "Instantaneous output growth", False),
        ("labor_share", "Labor share", False),
    ]

    for ax, (key, title, log_scale) in zip(axes.flat, panels):
        for color, (name, result) in zip(colors, results.items()):
            values = result[key]
            if key in {"output", "a"}:
                values = values / values[0]
            ax.plot(result["t"], values, lw=2.0, color=color, label=name)
        if log_scale:
            ax.set_yscale("log")
        if key == "output_growth":
            ax.yaxis.set_major_formatter(lambda value, _: f"{100 * value:.1f}%")
        ax.set_title(title)
        ax.set_xlabel("Years")
        ax.grid(alpha=0.25)

    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.savefig(FIGURE_DIR / "illustrative-regimes.pdf", bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "illustrative-regimes.png", dpi=220, bbox_inches="tight")

    for name, result in results.items():
        print(
            name,
            f"Y(80)/Y(0)={result['output'][-1] / result['output'][0]:.3f}",
            f"A(80)/A(0)={result['a'][-1] / result['a'][0]:.3f}",
            f"labor share(80)={result['labor_share'][-1]:.3f}",
        )


if __name__ == "__main__":
    main()