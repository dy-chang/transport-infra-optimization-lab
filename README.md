# Transportation Infrastructure Optimization Lab

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Optimization: SciPy HiGHS](https://img.shields.io/badge/Solver-HiGHS%20Branch--and--Cut%20MIP-darkblue.svg)](https://highs.dev/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Interactive%20Workbench-FF4B4B.svg)](https://streamlit.io/)
[![PyDeck](https://img.shields.io/badge/PyDeck-3D%20Spatial%20GIS-green.svg)](https://deckgl.readthedocs.io/)
[![Plotly](https://img.shields.io/badge/Plotly-Pareto%20Frontier-teal.svg)](https://plotly.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)
[![Tests: 100% Passing](https://img.shields.io/badge/Tests-6%20Passed%20(100%25)-success.svg)](tests/)

> **Multi-Objective Mixed-Integer Programming & Spatial Facility Location Lab**  
> Operations Research, Transportation Planning, and GIS integration solving capital allocation problems across **EV Charging (NEVI)**, **Transit Stops (FTA Title VI)**, **Vision Zero Safety (SS4A)**, and **Bike Infrastructure**.

---

## 1. Overview & Problem Context

Public transportation agencies, State DOTs, and MPOs face complex capital programming challenges: allocating limited capital budgets across thousands of candidate project locations to maximize community mobility while satisfying spatial separation, budget constraints, and federal equity requirements (e.g., the White House Justice40 Initiative).

This laboratory formulates and solves the **Multi-Objective Maximum Coverage Location Problem with Equity Quotas (MCLP-EQ)**. It bridges mathematical operations research with real-world transportation planning through:

1. **Dual-Engine Solvers**:
   - **Exact Mathematical Programming**: Global branch-and-cut optimization powered by native **HiGHS** (`scipy.optimize.milp`).
   - **Submodular Greedy Heuristic**: Ultra-fast marginal return optimization executing in **<30 ms** for instantaneous 60fps web slider interactivity.
2. **Four Extensible Transportation Sectors**:
   - ⚡ **EV Charging Infrastructure** (NEVI Formula Program & Grid Capacity)
   - 🚌 **Transit Stop / BRT Investment** (FTA Title VI & Zero-Car Household Access)
   - 🛡️ **Vision Zero Safety Countermeasures** (High Injury Network & Crash Reductions)
   - 🚲 **Protected Bike Corridors** (Level of Traffic Stress & Complete Streets)
3. **Real-World Open Data Case Study**:
   - Calibrated using Florida corridor networks (Orange, Hillsborough, Pinellas, and Southwest Florida).
   - **2,413 candidate parcels** conflated with real US Census socio-demographics, AADT traffic counts, and CEJST Justice40 Disadvantaged Community (DAC) tracts.

---

## 2. Mathematical Formulation

$$\max_{x, y} \quad Z = (1 - w_{\text{eq}}) \sum_{i \in \mathcal{I}} D_i y_i + w_{\text{eq}} \cdot \gamma \sum_{i \in \mathcal{I}_{\text{DAC}}} D_i y_i - \lambda \sum_{j \in \mathcal{J}} C_j x_j$$

**Subject to:**

1. **Capital Budget Constraint**:
   $$\sum_{j \in \mathcal{J}} C_j x_j \le B$$

2. **Cardinality Upper Bound**:
   $$\sum_{j \in \mathcal{J}} x_j \le K_{\max}$$

3. **Coverage Coupling Constraints**:
   $$y_i - \sum_{j \in \mathcal{N}(i)} x_j \le 0 \quad \forall i \in \mathcal{I}$$

4. **Anti-Cannibalization Spatial Spacing**:
   $$x_j + x_k \le 1 \quad \forall (j, k) \in \mathcal{E}_{\text{conflict}} \text{ where } \text{dist}(j, k) < d_{\min}$$

5. **Linear Justice40 Equity Quota Mandate**:
   $$\sum_{i \notin \mathcal{I}_{\text{DAC}}} \left( \alpha_{\text{equity\_min}} D_i \right) y_i - \sum_{i \in \mathcal{I}_{\text{DAC}}} \left( (1 - \alpha_{\text{equity\_min}}) D_i \right) y_i \le 0$$

6. **Integrality**:
   $$x_j \in \{0, 1\} \quad \forall j \in \mathcal{J}, \qquad 0 \le y_i \le 1 \quad \forall i \in \mathcal{I}$$

---

## 3. Interactive Web Workbench (`app.py`)

Run the interactive dashboard to explore multi-objective trade-offs in real time:

```bash
streamlit run app.py
```

### Key Interactive Features
- **Dynamic Sliders**:
  - `Budget`: `$2M ──●── $25M`
  - `Equity Priority Weight`: `Low ──●── High`
  - `Justice40 Quota`: `0% ──●── 50%`
  - `Sector Selector`: Switch between EV Charging, Transit Stops, Safety, and Bike Networks.
- **PyDeck 3D Geospatial Map**:
  - Selected facilities displayed as 3D columns (elevation proportional to demand).
  - Unfunded candidate pool rendered as subdued gray dots.
  - Census tracts color-coded by Justice40 Disadvantaged Community status (Cyan DAC vs Orange Non-DAC).
- **Pareto Trade-Off Curve**:
  - Automatically sweeps equity weights $[0.0, 1.0]$ to trace the empirical Pareto Frontier between System Efficiency and Equity.
- **Operations Research Benchmark Tab**:
  - Side-by-side comparison between HiGHS Branch-and-Cut and Submodular Greedy with runtime, optimality gap, and spatial agreement percentages.
- **Capital Program Export**:
  - Download the final funded project portfolio directly as a CSV capital improvement plan.

---

## 4. Multi-Sector Domain Profiles

| Domain Sector | Demand Metric ($D_i$) | Candidate Pool ($\mathcal{J}$) | Unit Cost Range ($C_j$) | Equity Priority ($\mathcal{I}_{\text{DAC}}$) | Federal Program |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **EV Charging Deployment** | Traffic AADT & Commercial Density | 2,413 Highway & Commercial Parcels | \$150k – \$320k | Justice40 Disadvantaged Communities | NEVI Formula (23 CFR Part 680) |
| **Transit / BRT Stops** | Transit Propensity & Activity Density | 1,850 Intersections & Hubs | \$50k – \$140k | Title VI Zero-Car Household Areas | FTA Capital Grants |
| **Vision Zero Safety** | 5-Year Fatal & Severe Injury Crashes | 1,620 High Injury Network Sites | \$80k – \$250k | Vulnerable Road Users (VRU) | Safe Streets for All (SS4A) |
| **Protected Bike Corridors** | Active Mobility Potential & LTS Gap | 2,100 Street Segments | \$70k – \$180k | Low-Income Car-Free Sheds | Reconnecting Communities |

---

## 5. Repository Structure

```text
transport-infra-optimization-lab/
├── README.md                          # Repository documentation
├── pyproject.toml                     # Package configuration and build rules
├── requirements.txt                   # Locked dependencies
├── .gitignore
├── app.py                             # Interactive Streamlit + PyDeck workbench
├── src/
│   ├── __init__.py
│   ├── config.py                      # Domain definitions, bounds, Justice40 quotas
│   ├── models.py                      # Pydantic & dataclass schemas
│   ├── data_loader.py                 # Real Florida corridor loader & candidate generator
│   ├── analytics.py                   # Optimality gap, Justice40 compliance, budget analytics
│   └── solvers/
│       ├── __init__.py
│       ├── base.py                    # BaseSolver abstract class
│       ├── highs_solver.py            # Exact HiGHS branch-and-cut MIP solver
│       ├── greedy_solver.py           # Submodular marginal return heuristic (<30ms)
│       └── pareto_frontier.py         # Multi-objective Pareto Frontier analyzer
├── tests/
│   ├── __init__.py
│   └── test_optimization.py           # Pytest suite (solvers, constraints, domains, Pareto)
├── docs/
│   └── mathematical_formulation.md    # Formal Operations Research & MIP specification
└── examples/
    └── run_pareto_analysis.py         # CLI script generating multi-sector Pareto curves
```

---

## 6. Quickstart & Usage

### 6.1 Installation
```bash
git clone git@github.com:dy-chang/transport-infra-optimization-lab.git
cd transport-infra-optimization-lab

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .
```

### 6.2 Running the Pareto Frontier CLI
```bash
python examples/run_pareto_analysis.py
```

### 6.3 Launching the Interactive Workbench
```bash
streamlit run app.py
```

### 6.4 Running Automated Tests
```bash
pytest tests/ -v
```

---

## 7. License
Distributed under the MIT License. See `LICENSE` for details.
