# Scalable Intelligence, Productivity, and Capital Accumulation

Replication package and simulation code for the manuscript:

> *"Scalable Intelligence, Productivity, and Capital Accumulation."*  
> *Manuscript under revision at Economics of Innovation and New Technology.*

---

## Overview

This repository contains the simulation code, replication scripts, empirical summary tables, and generated figures for the paper. The framework investigates the macroeconomic consequences of artificial intelligence (AI) when machine cognition is produced jointly by non-rival AI capability and reproducible compute capital, separating substitution, accumulation, research feedback, congestion, and distributional channels.

---

## Requirements and Tested Environment

The simulation scripts are written in standard Python 3 and require standard scientific computing packages. The full replication suite was tested and verified under:

- **Python**: `3.13.11` (compatible with standard Python 3.10+)
- **NumPy**: `2.2.6`
- **SciPy**: `1.16.3`
- **Matplotlib**: `3.10.8`
- **pandas**: `2.3.3` (used by `build_stylized_facts.py`)

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

### 2. Transition Robustness Table (Section 11.5, Table 1)

```bash
python scripts/feedback_robustness.py
```

- **Requirements**: NumPy, SciPy, and Matplotlib (via `distribution_feedback.py`).
- **Outputs**:
  - Summary dataset: `data/feedback_robustness_summary.csv` (and identical `data/feedback_robustness_summary.txt`, 16 rows)
- **Methodology & Interpretation**:
  - Reproduces the compact transition robustness analysis in Table 1 (Section 11.5).
  - Compares all 16 combinations of four transfer rates ($\tau \in \{0.0, 0.3, 0.6, 0.9\}$), two saving-propensity pairs ($(s_W, s_O) \in \{(0.15, 0.40), (0.25, 0.30)\}$), and two capability-feedback exponents ($\lambda \in \{0.15, 0.70\}$ with $\phi = 0.30$).
  - Evaluates terminal outcomes at year 80 (1,601 evaluation dates each) starting from common initial stocks ($K_0 = 4$, $X_0 = A_0 = L_0 = 1$, $\sigma = 2$, $\beta = 9/17$).
  - Verifies solver convergence to relative tolerance $10^{-9}$ (checked against $10^{-11}$), interior growth derivative errors ($< 1.1 \times 10^{-7}$), equal-propensity invariance across transfer rates, and the complementary unequal-depreciation capital ratio formula $\frac{\beta}{1-\beta}\frac{n+\delta_X}{n+\delta_K}$.
  - **Disclaimer**: These exercises illustrate finite-horizon feedback mechanisms, not calibrated forecasts or asymptotic regime switches.

### 3. Baseline Regimes Simulation

```bash
python scripts/simulate_regimes.py
```

- **Outputs**: `figures/illustrative-regimes.png` and `figures/illustrative-regimes.pdf` (Figure 2).
- Simulates three illustrative scenarios: labor complementarity, labor substitution, and strong AI-research feedback with congestion. These finite-horizon simulations do not establish finite-time escape.

### 4. Sensitivity Analysis

```bash
python scripts/sensitivity_analysis.py
```

- **Outputs**: `figures/sensitivity-map.png` and `figures/sensitivity-map.pdf` (Figure 3).
- Performs robustness sweeps over the elasticity of substitution ($\sigma$) and research-productivity parameters.

### 5. Source-Reported AI Scaling Trends

```bash
python scripts/build_stylized_facts.py
```

- **Inputs**: `data/source_reported_ai_scaling.csv`
- **Outputs**: `figures/ai-scaling-facts.png` and `figures/ai-scaling-facts.pdf` (Figure 1).
- Compiles published headline scaling metrics and citations into summary visualizations.

---

## Data Dictionaries

### 1. `data/distribution_feedback_paths.csv`

The continuous trajectory simulation dataset evaluated at 0.05-year steps over $t \in [0, 80]$ (1,601 rows per scenario, 6,404 data rows total):

| Column | Description | Format / Units |
| :--- | :--- | :--- |
| `scenario` | Scenario identifier specifying saving propensities ($s_W, s_O$) and transfer rate ($\tau$) | Text string |
| `time` | Simulation time step | Model years ($0.00$ to $80.00$) |
| `capital` | Conventional physical capital stock ($K_t$) | Index level ($K_0 = 4.0$) |
| `compute` | Cognitive compute capital stock ($X_t$) | Index level ($X_0 = 1.0$) |
| `capability` | Algorithmic AI capability level ($A_t$) | Index level ($A_0 = 1.0$) |
| `labor` | Human labor supply ($L_t = L_0 e^{nt}$) | Index level ($L_0 = 1.0$) |
| `output` | Aggregate output ($Y_t$) | Model output units |
| `saving` | Aggregate saving rate ($s_t$) | **Fraction** ($0$ to $1$) |
| `labor_share` | Human labor income share ($s_{L,t}$) | **Fraction** ($0$ to $1$) |
| `output_growth` | Instantaneous annual growth rate of output ($\dot{Y}_t / Y_t$) | **Fraction per year** (e.g., $0.0172 = 1.72\%$) |
| `consumption_wage` | Aggregate consumption of wage/transfer recipients ($\mathcal{C}_{W,t}$) | Output units |
| `consumption_capital` | Aggregate consumption of capital owners ($\mathcal{C}_{O,t}$) | Output units |

### 2. `data/feedback_robustness_summary.csv`

Terminal transition outcomes evaluated at model year $t = 80$ for Table 1 across all 16 robustness cases:

| Column | Description | Format / Units |
| :--- | :--- | :--- |
| `research_case` | Research feedback regime (`baseline` with $\lambda=0.15$ or `higher_capability_feedback` with $\lambda=0.70$) | Text string |
| `lam` | Capability-feedback exponent in research ($\lambda$) | Float (`0.15` or `0.70`) |
| `phi` | Effective-research-input exponent ($\phi$) | Float (`0.30`) |
| `saving_wage` | Saving propensity of wage/transfer recipients ($s_W$) | Float (`0.15` or `0.25`) |
| `saving_capital` | Saving propensity of capital owners ($s_O$) | Float (`0.40` or `0.30`) |
| `transfer` | Nonlabor gross income transfer fraction ($\tau$) | Float (`0.0`, `0.3`, `0.6`, `0.9`) |
| `time` | Evaluation time horizon | Model years (`80.0`) |
| `capital` | Terminal conventional physical capital stock ($K_{80}$) | Model index level |
| `compute` | Terminal cognitive compute capital stock ($X_{80}$) | Model index level |
| `capability` | Terminal algorithmic AI capability level ($A_{80}$) | Model index level |
| `output` | Terminal aggregate output ($Y_{80}$) | Model output units |
| `saving` | Terminal aggregate saving rate ($s_{80}$) | **Fraction** ($0$ to $1$) |
| `labor_share` | Terminal labor income share ($s_{L,80}$) | **Fraction** ($0$ to $1$) |
| `output_growth` | Instantaneous annual output growth rate ($\dot{Y}_{80}/Y_{80}$) | **Fraction per year** (multiply by 100 for Table 1 %) |

*Note: For environments where CSV downloads are constrained, `data/feedback_robustness_summary.txt` provides an identical comma-separated text representation.*

---

## Repository Structure

```text
├── data/
│   ├── distribution_feedback_paths.csv     # Generated simulation paths (4 scenarios, 6,404 rows)
│   ├── feedback_robustness_summary.csv     # Terminal transition outcomes for Table 1 (16 rows)
│   ├── feedback_robustness_summary.txt     # Identical comma-separated alternative
│   └── source_reported_ai_scaling.csv      # Source-reported AI training and inference scaling trends
├── figures/
│   ├── ai-scaling-facts.png (.pdf)          # Figure 1: Selected AI scaling trends
│   ├── illustrative-regimes.png (.pdf)      # Figure 2: Illustrative growth regimes
│   ├── sensitivity-map.png (.pdf)           # Figure 3: Sensitivity across elasticity and research scale
│   └── distribution-feedback.png (.pdf)     # Figure 4: Distribution and saving feedback paths
├── scripts/
│   ├── build_stylized_facts.py              # Generates Figure 1 from reported scaling data
│   ├── distribution_feedback.py             # Simulates distribution feedback, verifies tolerances, generates Figure 4 & CSV
│   ├── feedback_robustness.py               # Reproduces Table 1 transition robustness summary & checks
│   ├── sensitivity_analysis.py              # Generates sensitivity map (Figure 3)
│   └── simulate_regimes.py                  # Core regime simulation model (Figure 2)
├── .gitignore                               # Git ignore rules for python cache and local files
└── README.md                                # Replication documentation and guide
```

---

## Validation and Numerical Checks

`scripts/distribution_feedback.py` and `scripts/feedback_robustness.py` perform automated internal validation checks:
1. **Positivity and finite values**: Guaranteed across all simulated stocks ($K, X, A, L, Y$).
2. **Accounting balance**: $\mathcal{C}_W + \mathcal{C}_O + sY = Y$ verified at every evaluation date.
3. **Factor share bounds**: $s_{L,t} \in [0, 1 - \alpha]$ and aggregate saving bounded by the convex combination of saving propensities.
4. **Analytical derivative consistency**: Analytical growth $\dot{Y}/Y$ matches numerical centered differences of $\log(Y)$ within $2 \times 10^{-5}$ along all paths (and $< 1.1 \times 10^{-7}$ in the robustness table).
5. **Tolerance convergence**: Solutions remain invariant to within $< 2 \times 10^{-9}$ when relative tolerance is tightened from $10^{-9}$ to $10^{-11}$.
6. **Equal-propensity invariance**: Redistribution across transfer rates produces identical aggregate paths when $s_W = s_O$ (verified to relative error $< 10^{-8}$).
7. **Limiting analytical checks**: Complementary bottleneck steady-state ratios and large-stock policy threshold directions match analytical formulas.
