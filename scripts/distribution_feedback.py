"""Reproduce and validate the revision's distribution and policy experiments."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from simulate_regimes import COMMON, SCENARIOS, levels_and_flows, simulate


ROOT = Path(__file__).resolve().parents[1]
BETA = COMMON["s_k"] / (COMMON["s_k"] + COMMON["s_x"])
BASE = COMMON | SCENARIOS["Substitution ($\\sigma=2$)"]


def income_accounts(labor_share, saving_wage, saving_capital, transfer):
    recipient_share = labor_share + transfer * (1.0 - labor_share)
    capital_share = (1.0 - transfer) * (1.0 - labor_share)
    saving = saving_wage * recipient_share + saving_capital * capital_share
    consumption_wage = (1.0 - saving_wage) * recipient_share
    consumption_capital = (1.0 - saving_capital) * capital_share
    np.testing.assert_allclose(recipient_share + capital_share, 1.0, atol=1e-14)
    np.testing.assert_allclose(saving + consumption_wage + consumption_capital, 1.0, atol=1e-14)
    return saving, consumption_wage, consumption_capital


def run_path(saving_wage, saving_capital, transfer, parameters=None,
             horizon=80.0, points=1601, initial=(4.0, 1.0, 1.0),
             frozen_capability=False, tolerance=1e-9):
    if not (0 < saving_wage < 1 and 0 < saving_capital < 1 and 0 <= transfer <= 1):
        raise ValueError("Invalid saving propensity or transfer fraction")
    parameters = BASE | (parameters or {})

    def derivative(time, state):
        capital, compute, capability, _, output, research, labor_share = levels_and_flows(
            time, state, parameters)
        saving, _, _ = income_accounts(labor_share, saving_wage, saving_capital, transfer)
        return np.array([BETA * saving * output / capital - parameters["delta_k"],
                         (1.0 - BETA) * saving * output / compute - parameters["delta_x"],
                         0.0 if frozen_capability else research / capability])

    times = np.linspace(0, horizon, points)
    solution = solve_ivp(derivative, (0, horizon), np.log(initial), t_eval=times,
                         rtol=tolerance, atol=tolerance / 100)
    if not solution.success or solution.t.size != times.size:
        raise RuntimeError(solution.message)
    flows = np.array([levels_and_flows(time, state, parameters)
                      for time, state in zip(times, solution.y.T)])
    capital, compute, capability, labor, output, _, labor_share = flows.T
    saving, consumption_wage, consumption_capital = income_accounts(
        labor_share, saving_wage, saving_capital, transfer)
    growth = np.array([derivative(time, state)
                       for time, state in zip(times, solution.y.T)])
    human_weight = labor_share / (1.0 - parameters["alpha"])
    output_growth = parameters["alpha"] * growth[:, 0] + (1.0 - parameters["alpha"]) * (
        human_weight * parameters["n"] + (1.0 - human_weight) * (growth[:, 1] + growth[:, 2]))
    finite_difference = np.gradient(np.log(output), times, edge_order=2)
    derivative_error = float(np.max(np.abs(finite_difference[1:-1] - output_growth[1:-1])))
    if derivative_error > 2e-5:
        raise AssertionError(f"Growth derivative error: {derivative_error}")
    for values in (capital, compute, capability, labor, output):
        if not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise AssertionError("Nonpositive or nonfinite simulated level")
    if np.any(labor_share < 0) or np.any(labor_share > 1 - parameters["alpha"] + 1e-12):
        raise AssertionError("Invalid labor share")
    if np.any(saving < min(saving_wage, saving_capital) - 1e-12) or np.any(
            saving > max(saving_wage, saving_capital) + 1e-12):
        raise AssertionError("Saving violates its convex-combination bound")
    return {"time": times, "capital": capital, "compute": compute,
            "capability": capability, "labor": labor, "output": output,
            "saving": saving, "labor_share": labor_share, "output_growth": output_growth,
            "consumption_wage": consumption_wage * output,
            "consumption_capital": consumption_capital * output,
            "derivative_error": derivative_error}


def main():
    cases = {"Different propensities, transfer = 0": (0.15, 0.40, 0.0),
             "Different propensities, transfer = 0.9": (0.15, 0.40, 0.9),
             "Equal propensities, transfer = 0": (0.34, 0.34, 0.0)}
    results = {label: run_path(*settings) for label, settings in cases.items()}
    for label, settings in cases.items():
        tighter = run_path(*settings, tolerance=1e-11)
        relative_error = max(float(np.max(np.abs(results[label][key] / tighter[key] - 1)))
                             for key in ("capital", "compute", "capability", "output"))
        if relative_error > 1e-6:
            raise AssertionError("Tolerance convergence check failed")
        print(label, "relative tolerance error", relative_error,
              "growth derivative error", results[label]["derivative_error"])
    equal_transfer = run_path(0.34, 0.34, 0.9)
    equal_no_transfer = results["Equal propensities, transfer = 0"]
    reference = simulate(SCENARIOS["Substitution ($\\sigma=2$)"], points=1601)
    for key, old_key in (("capital", "k"), ("compute", "x"), ("capability", "a"),
                         ("output", "output")):
        np.testing.assert_allclose(equal_transfer[key], equal_no_transfer[key], rtol=1e-8)
        np.testing.assert_allclose(equal_no_transfer[key], reference[old_key], rtol=1e-8)
    common_depreciation = run_path(0.15, 0.4, 0.9,
                                   parameters={"delta_k": 0.08, "delta_x": 0.08})
    imbalance = (1 - BETA) * common_depreciation["capital"] - BETA * common_depreciation["compute"]
    np.testing.assert_allclose(imbalance, imbalance[0] * np.exp(-0.08 * common_depreciation["time"]),
                               rtol=2e-5, atol=2e-8)
    bottleneck = run_path(0.15, 0.4, 0.9, parameters={"sigma": 0.6},
                          initial=(4.0, 1.0, 1e12), horizon=600, points=6001,
                          frozen_capability=True)
    bottleneck_saving = 0.15 * (1 - BASE["alpha"]) + BASE["alpha"] * (0.15 * 0.9 + 0.4 * 0.1)
    bottleneck_exponent = (0.6 - 1) / 0.6
    bottleneck_productivity = (BASE["omega"] ** (1 / bottleneck_exponent)
                               * (1 - BASE["ell_r"])) ** (1 - BASE["alpha"])
    steady_capital = (BETA * bottleneck_saving * bottleneck_productivity /
                      (BASE["n"] + BASE["delta_k"])) ** (1 / (1 - BASE["alpha"]))
    steady_output = bottleneck_productivity * steady_capital ** BASE["alpha"]
    steady_compute = (1 - BETA) * bottleneck_saving * steady_output / (BASE["n"] + BASE["delta_x"])
    np.testing.assert_allclose(bottleneck["capital"][-1] / bottleneck["labor"][-1],
                               steady_capital, rtol=1e-6)
    np.testing.assert_allclose(bottleneck["compute"][-1] / bottleneck["labor"][-1],
                               steady_compute, rtol=1e-6)
    alpha = BASE["alpha"]
    exponent = (BASE["sigma"] - 1) / BASE["sigma"]
    coefficient = ((1 - BASE["omega"]) ** ((1 - alpha) / exponent)
                   * (8 * (1 - BASE["u_r"])) ** (1 - alpha)
                   * BETA ** alpha * (1 - BETA) ** (1 - alpha))
    threshold = (0.40 - (BASE["n"] + 0.08) / coefficient) / (0.40 - 0.15)
    print("Frozen A=8, common depreciation=0.08: coefficient", coefficient,
          "critical transfer", threshold)
    for transfer in (0.0, 0.9):
        asymptotic_growth = (0.4 - 0.25 * transfer) * coefficient - 0.08
        print("Transfer", transfer, "limiting large-stock capital growth", asymptotic_growth)
        frozen = run_path(0.15, 0.4, transfer, parameters={"delta_k": 0.08, "delta_x": 0.08},
                          initial=(BETA * 1e6, (1 - BETA) * 1e6, 8),
                          horizon=400, points=4001, frozen_capability=True)
        growth_per_worker = np.log((frozen["compute"][-1] / frozen["labor"][-1]) /
                                  frozen["compute"][0])
        if np.sign(growth_per_worker) != np.sign(asymptotic_growth - BASE["n"]):
            raise AssertionError("Large-stock policy threshold check failed")
    figure_directory = ROOT / "figures"
    figure_directory.mkdir(exist_ok=True)
    figure, axes = plt.subplots(2, 3, figsize=(11.4, 6.7), constrained_layout=True)
    panels = [("saving", "Aggregate saving rate", False),
              ("compute", "Compute / initial compute", True),
              ("capability", "Capability / initial capability", True),
              ("output_growth", "Output growth (% per year)", False),
              ("labor_share", "Labor share", False),
              ("consumption_wage", "Wage/transfer consumption / initial output", False)]
    colors = ["#245a81", "#c66a18", "#4c956c"]
    for axis, (key, title, logarithmic) in zip(axes.flat, panels):
        for color, (label, result) in zip(colors, results.items()):
            values = result[key].copy()
            if logarithmic:
                values /= values[0]
            if key == "output_growth":
                values *= 100
            if key == "consumption_wage":
                values /= result["output"][0]
            axis.plot(result["time"], values, label=label, color=color)
        if key == "consumption_wage":
            axis.plot(equal_transfer["time"], equal_transfer[key] / equal_transfer["output"][0],
                      color=colors[2], linestyle="--", label="Equal propensities, transfer = 0.9")
        if logarithmic:
            axis.set_yscale("log")
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("Model years")
        axis.grid(alpha=0.2)
    handles, labels = axes.flat[-1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=9)
    figure.savefig(figure_directory / "distribution-feedback.pdf", bbox_inches="tight")
    figure.savefig(figure_directory / "distribution-feedback.png", dpi=200, bbox_inches="tight")
    plt.close(figure)
    with (ROOT / "data" / "distribution_feedback_paths.csv").open("w", newline="") as stream:
        fields = [key for key in equal_no_transfer if key != "derivative_error"]
        writer = csv.writer(stream)
        writer.writerow(["scenario"] + fields)
        exported_results = results | {"Equal propensities, transfer = 0.9": equal_transfer}
        for label, result in exported_results.items():
            writer.writerows([label] + [result[key][index] for key in fields]
                             for index in range(len(result["time"])))
    for label, result in results.items():
        print(label, {key: float(result[key][-1]) for key in
                      ("saving", "compute", "capability", "output_growth", "labor_share")})


if __name__ == "__main__":
    main()