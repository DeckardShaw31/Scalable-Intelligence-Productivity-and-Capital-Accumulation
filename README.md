# Scalable Intelligence, Productivity, and Capital Accumulation

Replication package and simulation code for the manuscript:

> **Nguyen, N. T.** *"Scalable Intelligence, Productivity, and Capital Accumulation."* *Economics of Innovation and New Technology*.

---

## Overview

This repository contains the simulation code, replication scripts, empirical summary tables, and generated figures for the paper. The framework investigates the macroeconomic consequences of artificial intelligence (AI) when machine cognition is produced jointly by non-rival AI capability and reproducible compute capital, separating substitution, accumulation, research feedback, congestion, and distributional channels.

---

## Requirements

The simulation scripts are written in standard Python 3 and require the following scientific computing packages:

- `numpy`
- `scipy`
- `matplotlib`
- `pandas` (for stylized facts compilation)

Install dependencies via `pip`:

```bash
pip install numpy scipy matplotlib pandas
```

---

## Replication Instructions

All scripts should be executed from the repository root directory.

### 1. Saving-Feedback and Distribution Experiments

```bash
python scripts/distribution_feedback.py
```

- **Requirements**: NumPy, SciPy, and Matplotlib.
- **Outputs**:
  - Figure 4: `figures/distribution-feedback.png` (and `figures/distribution-feedback.pdf`)
  - Simulation dataset: `data/distribution_feedback_paths.csv` (4 scenarios, 6,404 rows)
- **Methodology & Interpretation**:
  - Replaces fixed saving rates with a distribution-sensitive saving rule and balanced transfer parameter ($\tau$).
  - Simulates four scenarios over an 80-year horizon (1,601 evaluation dates each):
    1. *Different propensities, transfer = 0* ($s_W = 0.15$, $s_O = 0.40$, $\tau = 0.0$)
    2. *Different propensities, transfer = 0.9* ($s_W = 0.15$, $s_O = 0.40$, $\tau = 0.9$)
    3. *Equal propensities, transfer = 0* ($s_W = 0.34$, $s_O = 0.34$, $\tau = 0.0$)
    4. *Equal propensities, transfer = 0.9* ($s_W = 0.34$, $s_O = 0.34$, $\tau = 0.9$)
  - **Important Note on Data Format**: In `data/distribution_feedback_paths.csv`, growth rates (such as `output_growth`) and shares (such as `labor_share` and `saving`) are recorded as **fractions**, not percentages (e.g., $0.02$ represents $2\%$).
  - **Disclaimer**: These exercises use synthetic, illustrative scenarios designed to isolate analytical mechanisms and comparative dynamics—they are not empirical estimates, econometric forecasts, or policy recommendations.

### 2. Baseline Regimes Simulation

```bash
python scripts/simulate_regimes.py
```

- **Outputs**: `figures/illustrative-regimes.png` and `figures/illustrative-regimes.pdf` (Figure 2).
- Simulates illustrative growth trajectories across technological regimes (Solow bottleneck, AK limit, accelerating growth, and congested escape).

### 3. Sensitivity Analysis

```bash
python scripts/sensitivity_analysis.py
```

- **Outputs**: `figures/sensitivity-map.png` and `figures/sensitivity-map.pdf` (Figure 3).
- Performs robustness sweeps over the elasticity of substitution ($\sigma$) and research-productivity parameters.

### 4. Source-Reported AI Scaling Trends

```bash
python scripts/build_stylized_facts.py
```

- **Inputs**: `data/source_reported_ai_scaling.csv`
- **Outputs**: `figures/ai-scaling-facts.png` and `figures/ai-scaling-facts.pdf` (Figure 1).
- Compiles published headline scaling metrics and citations into summary visualizations.

---

## Repository Structure

```text
├── data/
│   ├── distribution_feedback_paths.csv   # Generated simulation data (4 scenarios, 6,404 rows)
│   └── source_reported_ai_scaling.csv    # Source-reported AI training and inference scaling trends
├── figures/
│   ├── ai-scaling-facts.png (.pdf)        # Figure 1: Selected AI scaling trends
│   ├── illustrative-regimes.png (.pdf)    # Figure 2: Illustrative growth regimes
│   ├── sensitivity-map.png (.pdf)         # Figure 3: Sensitivity across elasticity and research scale
│   └── distribution-feedback.png (.pdf)   # Figure 4: Distribution and saving feedback paths
├── scripts/
│   ├── build_stylized_facts.py            # Generates Figure 1 from reported scaling data
│   ├── distribution_feedback.py           # Simulates distribution feedback, verifies tolerances, generates Figure 4 & CSV
│   ├── sensitivity_analysis.py            # Generates sensitivity map (Figure 3)
│   └── simulate_regimes.py                # Core regime simulation model (Figure 2)
└── README.md                              # Replication documentation and guide
```

---

## Validation and Numerical Checks

`scripts/distribution_feedback.py` performs automated internal validation checks on every run:
1. Positivity and finite-value assertions for all capital, compute, capability, and output paths.
2. Factor share and aggregate saving convex-combination bounds.
3. Accounting balance: $\mathcal{C}_W + \mathcal{C}_O + sY = Y$.
4. Derivative consistency: analytical output growth matches interior finite differences of $\log(Y)$ within $2 \times 10^{-5}$.
5. Relative convergence tolerance verification tightening solver tolerances from $10^{-9}$ to $10^{-11}$.
6. Asymptotic policy threshold and bottleneck limiting checks.
