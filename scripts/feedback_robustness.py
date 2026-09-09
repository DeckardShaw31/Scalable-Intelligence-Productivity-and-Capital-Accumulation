"""Reproduce the compact transition robustness table; leave baseline outputs intact.

Run from the repository root: python scripts/feedback_robustness.py
The CSV and identical .txt alternative report terminal levels and annual growth
rates, not calibrated forecasts or tests of asymptotic regime switching.
"""

import csv
from pathlib import Path

import numpy as np

from distribution_feedback import BASE, BETA, run_path


def main():
    rows = []
    largest_error = 0.0
    largest_derivative_error = 0.0
    for label, lam in (("baseline", 0.15), ("higher_capability_feedback", 0.70)):
        for sw, so in ((0.15, 0.40), (0.25, 0.30)):
            for tau in (0.0, 0.3, 0.6, 0.9):
                path = run_path(sw, so, tau, parameters={"lam": lam})
                tight = run_path(sw, so, tau, parameters={"lam": lam}, tolerance=1e-11)
                error = max(float(np.max(np.abs(path[key] / tight[key] - 1)))
                            for key in ("capital", "compute", "capability", "output"))
                if error > 1e-6:
                    raise AssertionError("Tolerance convergence failed")
                largest_error = max(largest_error, error)
                largest_derivative_error = max(largest_derivative_error, path["derivative_error"])
                rows.append(dict(research_case=label, lam=lam, phi=BASE["phi"],
                                 saving_wage=sw, saving_capital=so, transfer=tau,
                                 time=float(path["time"][-1]),
                                 **{key: float(path[key][-1]) for key in
                                    ("capital", "compute", "capability", "output", "saving",
                                     "labor_share", "output_growth")}))
        null = run_path(0.34, 0.34, 0.0, parameters={"lam": lam})
        for tau in (0.3, 0.6, 0.9):
            transferred = run_path(0.34, 0.34, tau, parameters={"lam": lam})
            for key in ("capital", "compute", "capability", "output"):
                np.testing.assert_allclose(null[key], transferred[key], rtol=1e-8, atol=1e-10)

    # Regression check for the complementary, unequal-depreciation ratio.
    bottleneck = run_path(0.15, 0.40, 0.9, parameters={"sigma": 0.6},
                          initial=(4.0, 1.0, 1e12), horizon=600, points=6001,
                          frozen_capability=True)
    expected = BETA / (1 - BETA) * (BASE["n"] + BASE["delta_x"]) / (BASE["n"] + BASE["delta_k"])
    np.testing.assert_allclose(bottleneck["capital"][-1] / bottleneck["compute"][-1],
                               expected, rtol=1e-6)
    directory = Path(__file__).resolve().parents[1] / "data"
    directory.mkdir(exist_ok=True)
    for suffix in ("csv", "txt"):
        with (directory / f"feedback_robustness_summary.{suffix}").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    for row in rows:
        print(row["research_case"], row["saving_wage"], row["saving_capital"], row["transfer"],
              *(f"{row[key]:.3f}" for key in ("compute", "capability")),
              f'{100 * row["output_growth"]:.3f}', f'{row["labor_share"]:.3f}')
    print("Maximum relative tolerance error:", largest_error)
    print("Maximum growth derivative error:", largest_derivative_error)
    print("Complementary capital ratio:", expected)
    print("All checks passed; CSV and identical TXT summary saved in data/.")


if __name__ == "__main__":
    main()