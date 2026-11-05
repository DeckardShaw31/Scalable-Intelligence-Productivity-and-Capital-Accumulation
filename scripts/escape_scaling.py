"""Check the endogenous compute--capability scaling and escape-timing results.

Run from the repository root: python scripts/escape_scaling.py

The script integrates the full benchmark system (fixed saving) and the
distribution-sensitive saving extension with active research, entirely in
logarithms so that paths can be followed close to finite-time escape.  It checks

  * the derived scaling exponent mu* = (2 - alpha - lambda - phi + q_hat) / phi
    and the asymptotic capability-growth coefficient under substitution;
  * the complementary-case exponent mu_c = -eta / phi and growth rate
    phi * n / (-eta), with eta = lambda + phi - 1 - q_hat;
  * escape dates for the Figure 1 scenarios and, under distribution-sensitive
    saving, for alternative transfer rates.

Outputs are written to data/escape_scaling_summary.csv.  They are synthetic
model outputs, not forecasts.
"""

import csv
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from distribution_feedback import BASE, BETA
from simulate_regimes import COMMON, SCENARIOS


ROOT = Path(__file__).resolve().parents[1]
ESCAPE_LOG_A = 40.0   # escape date: first date with log A = 40
CHECK_LOG_A = 60.0    # robustness check of the escape date
COMMON_LOG_A = 12.0   # common capability level for remaining-time comparisons


def q_hat(par):
    return par["q"] if par["zeta"] > 0 else 0.0


def mu_star(par):
    return (2.0 - par["alpha"] - par["lam"] - par["phi"] + q_hat(par)) / par["phi"]


def b_coefficient(par):
    nu = (par["sigma"] - 1.0) / par["sigma"]
    return (1.0 - par["omega"]) ** ((1.0 - par["alpha"]) / nu) * (1.0 - par["u_r"]) ** (
        1.0 - par["alpha"])


def log_compute_services(t, lx, par):
    """Log compute services.  Without an essential input these equal log X.

    With an essential input E_t = E_0 exp(n_E t), services are
    Phi(X, E) = [1/X + 1/(phi_bar E)]^(-1), which is constant-returns, increasing,
    concave, and bounded by phi_bar * E.
    """
    if "e0" not in par:
        return lx
    l_cap = np.log(par["phi_bar"] * par["e0"]) + par["n_e"] * t
    return -np.logaddexp(-lx, -l_cap)


def log_flows(t, y, par):
    """Return log output, final-production labor share, and capability growth."""
    lk, lx_stock, la = y
    lx = log_compute_services(t, lx_stock, par)
    a, om, nu = par["alpha"], par["omega"], (par["sigma"] - 1.0) / par["sigma"]
    ll = par["n"] * t
    t1 = np.log(om) + nu * (np.log(1.0 - par["ell_r"]) + ll)
    t2 = np.log(1.0 - om) + nu * (la + np.log(1.0 - par["u_r"]) + lx)
    l_terms = np.logaddexp(t1, t2)
    l_cognition = l_terms / nu
    l_output = a * lk + (1.0 - a) * l_cognition
    labor_share = (1.0 - a) * np.exp(t1 - l_terms)
    l_research_input = np.logaddexp(np.log(par["ell_r"]) + ll if par["ell_r"] > 0 else -np.inf,
                                    np.log(par["xi"] * par["u_r"]) + la + lx)
    l_congestion = (np.logaddexp(0.0, np.log(par["zeta"]) + par["q"] * la)
                    if par["zeta"] > 0 else 0.0)
    g_a = par["chi"] * np.exp((par["lam"] - 1.0) * la + par["phi"] * l_research_input
                              - l_congestion)
    return l_output, labor_share, g_a


def make_rhs(par, saving_rule):
    def rhs(t, y):
        l_output, labor_share, g_a = log_flows(t, y, par)
        s_k, s_x = saving_rule(labor_share)
        return [s_k * np.exp(l_output - y[0]) - par["delta_k"],
                s_x * np.exp(l_output - y[1]) - par["delta_x"],
                g_a]
    return rhs


def integrate(par, saving_rule, final_log_a=CHECK_LOG_A, initial=(4.0, 1.0, 1.0)):
    """Integrate with log capability as the clock (capability is strictly increasing).

    The state is (calendar time, log K, log X) as a function of log A, so a
    finite-time escape in calendar time appears as a finite limit of t(log A).
    """
    rhs_t = make_rhs(par, saving_rule)
    a0 = np.log(initial[2])

    def rhs_s(s, z):
        t, lk, lx = z
        g_k, g_x, g_a = rhs_t(t, (lk, lx, s))
        return [1.0 / g_a, g_k / g_a, g_x / g_a]

    sol = solve_ivp(rhs_s, (a0, final_log_a), [0.0, np.log(initial[0]), np.log(initial[1])],
                    method="LSODA", rtol=1e-10, atol=1e-12, dense_output=True)
    if not sol.success:
        raise RuntimeError(sol.message)
    date = lambda s: float(sol.sol(s)[0])
    return sol, rhs_s, date(COMMON_LOG_A), date(ESCAPE_LOG_A), date(CHECK_LOG_A)


def fixed_saving(par):
    return lambda labor_share: (par["s_k"], par["s_x"])


def distribution_saving(s_w, s_o, tau):
    def rule(labor_share):
        s = s_w * (labor_share + tau * (1.0 - labor_share)) + s_o * (1.0 - tau) * (
            1.0 - labor_share)
        return BETA * s, (1.0 - BETA) * s
    return rule


def local_elasticity(sol, rhs_s, s):
    """d log X / d log A and K/X at log capability s."""
    z = sol.sol(s)
    return float(rhs_s(s, z)[2]), float(np.exp(z[1] - z[2]))


def main():
    rows = []
    print("Figure 1 scenarios under fixed saving")
    for name, overrides in SCENARIOS.items():
        par = COMMON | overrides
        eta = par["lam"] + par["phi"] - 1.0 - q_hat(par)
        if par["sigma"] > 1.0:
            mu = mu_star(par)
            sol, rhs_s, common, escape, check = integrate(par, fixed_saving(par))
            if check - escape > 1e-3:
                raise AssertionError("Escape date is not robust to the threshold")
            elasticity, ratio = local_elasticity(sol, rhs_s, ESCAPE_LOG_A)
            if abs(elasticity / mu - 1.0) > 2e-3:
                raise AssertionError(f"Scaling exponent check failed for {name}")
            if abs(ratio / (par["s_k"] / par["s_x"]) - 1.0) > 2e-3:
                raise AssertionError("Capital-ratio limit check failed")
            la = ESCAPE_LOG_A
            z = sol.sol(la)
            _, _, g_a = log_flows(z[0], (z[1], z[2], la), par)
            predicted = b_coefficient(par) * par["s_k"] ** par["alpha"] * par["s_x"] ** (
                1.0 - par["alpha"]) / mu
            coefficient = g_a / np.exp((1.0 - par["alpha"]) * la)
            if abs(coefficient / predicted - 1.0) > 5e-3:
                raise AssertionError("Capability-growth coefficient check failed")
            print(f"  {name}: mu*={mu:.3f}, simulated dlnX/dlnA={elasticity:.3f}, "
                  f"K/X={ratio:.4f} (s_K/s_X={par['s_k'] / par['s_x']:.4f}), "
                  f"growth coefficient {coefficient:.5f} vs {predicted:.5f}, "
                  f"escape year {escape:.2f} (log A=60 at {check:.2f})")
            rows.append(dict(case=name, saving="fixed", transfer="", mu_predicted=mu,
                             mu_simulated=elasticity, escape_year=escape,
                             year_log_a_12=common, growth_coefficient=coefficient,
                             growth_coefficient_predicted=predicted))
        else:
            mu_c = -eta / par["phi"]
            rate = par["phi"] * par["n"] / (-eta)
            sol, rhs_s, _, _, _ = integrate(par, fixed_saving(par))
            z = sol.sol(CHECK_LOG_A)
            t_end = float(z[0])
            elasticity = float(rhs_s(CHECK_LOG_A, z)[2])
            g_a = 1.0 / float(rhs_s(CHECK_LOG_A, z)[0])
            if abs(g_a / rate - 1.0) > 2e-2 or abs(elasticity / mu_c - 1.0) > 2e-2:
                raise AssertionError("Complementary-case limits not reached")
            print(f"  {name}: eta={eta:.3f}, mu_c={mu_c:.3f}, simulated dlnX/dlnA="
                  f"{elasticity:.3f}; g_A={g_a:.5f} vs phi*n/(-eta)={rate:.5f} "
                  f"at year {t_end:.0f}; no escape (exponential capability growth)")
            rows.append(dict(case=name, saving="fixed", transfer="", mu_predicted=mu_c,
                             mu_simulated=elasticity, escape_year="none"))

    print("Distribution-sensitive saving with active research (s_W=0.15, s_O=0.40)")
    for label, lam in (("baseline", 0.15), ("higher_capability_feedback", 0.70)):
        par = BASE | {"lam": lam}
        dates = {}
        for tau in (0.0, 0.3, 0.6, 0.9, 1.0):
            _, _, common, escape, check = integrate(par, distribution_saving(0.15, 0.40, tau))
            if check - escape > 1e-3:
                raise AssertionError("Escape date not robust under distribution-sensitive saving")
            dates[tau] = (common, escape)
            rows.append(dict(case=label, saving="distribution", transfer=tau,
                             mu_predicted=mu_star(par), escape_year=escape,
                             year_log_a_12=common))
        s_inf = {tau: 0.15 * tau + 0.40 * (1.0 - tau) for tau in dates}
        base_remaining = dates[0.0][1] - dates[0.0][0]
        print(f"  {label} (lambda={lam}):")
        for tau, (common, escape) in dates.items():
            remaining = escape - common
            print(f"    tau={tau:.1f}: escape year {escape:7.1f}; remaining time from "
                  f"log A=12 relative to tau=0: {remaining / base_remaining:.3f} "
                  f"(asymptotic prediction {s_inf[0.0] / s_inf[tau]:.3f})")
        # Equal propensities remove the transfer effect on the escape date.
        equal = [integrate(par, distribution_saving(0.34, 0.34, tau))[3] for tau in (0.0, 0.9)]
        if abs(equal[0] - equal[1]) > 1e-6:
            raise AssertionError("Equal-propensity escape dates differ")

    print("Essential input in compute services (Corollary 1), strong-feedback case")
    strong = COMMON | SCENARIOS["Strong feedback + congestion"]
    essential = {"e0": 50.0, "phi_bar": 1.0, "n_e": 0.0}
    par = strong | essential
    eta = par["lam"] + par["phi"] - 1.0 - q_hat(par)
    bound = par["phi"] * max(par["n"], par["n_e"]) / (-eta)
    rhs = make_rhs(par, fixed_saving(par))
    sol = solve_ivp(rhs, (0.0, 3000.0), np.log((4.0, 1.0, 1.0)), method="LSODA",
                    rtol=1e-10, atol=1e-12)
    if not sol.success or sol.t[-1] < 3000.0:
        raise AssertionError("Essential-input path did not remain finite")
    g_a_end = rhs(sol.t[-1], sol.y[:, -1])[2]
    if not g_a_end <= bound:
        raise AssertionError("Capability growth exceeds the Corollary 1 bound")
    print(f"  q=1 (eta={eta:.2f}): no escape through year 3000; log A(3000)="
          f"{sol.y[2, -1]:.2f}; g_A(3000)={g_a_end:.5f} <= bound {bound:.5f}")
    rows.append(dict(case="strong feedback, essential input", saving="fixed", transfer="",
                     escape_year="none", capability_growth_year_3000=g_a_end,
                     capability_growth_bound=bound))
    par_low_q = par | {"q": 0.2}
    eta_low = par_low_q["lam"] + par_low_q["phi"] - 1.0 - q_hat(par_low_q)
    _, _, _, escape, check = integrate(par_low_q, fixed_saving(par_low_q))
    if eta_low <= 0 or check - escape > 1e-3:
        raise AssertionError("Escape with eta > 0 not confirmed")
    print(f"  q=0.2 (eta={eta_low:.2f}): escape year {escape:.1f}")
    rows.append(dict(case="strong feedback, essential input, q=0.2", saving="fixed",
                     transfer="", escape_year=escape))

    out = ROOT / "data" / "escape_scaling_summary.csv"
    keys = sorted({key for row in rows for key in row})
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
