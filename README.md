# Transportation Infrastructure Optimization Lab

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Optimization: SciPy HiGHS](https://img.shields.io/badge/Solver-HiGHS%20Branch--and--Cut%20MILP-darkblue.svg)](https://highs.dev/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Interactive%20Workbench-FF4B4B.svg)](https://streamlit.io/)
[![PyDeck](https://img.shields.io/badge/PyDeck-3D%20Spatial%20GIS-green.svg)](https://deckgl.readthedocs.io/)
[![Plotly](https://img.shields.io/badge/Plotly-Pareto%20Frontier-teal.svg)](https://plotly.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)
[![Tests: 100% Passing](https://img.shields.io/badge/Tests-10%20Passed%20(100%25)-success.svg)](tests/)

> **Where should limited transportation investment go?**  
> An open-source decision-support framework for optimizing transportation infrastructure investments under budget, coverage, spatial, and equity constraints.

---

### Four Planning Applications
`⚡ EV Charging` &emsp;|&emsp; `🚌 Transit Stops` &emsp;|&emsp; `🛡️ Safety Countermeasures` &emsp;|&emsp; `🚲 Bike Infrastructure`

---

## 1. Executive Summary

Public transportation agencies and MPOs face capital programming challenges: allocating limited capital budgets across thousands of candidate project locations to maximize mobility benefit while satisfying spatial separation, budget constraints, and equity goals.

This laboratory implements a **Multi-Objective Maximum Coverage Location Problem (MCLP)** framework with a **linear equity-share constraint**. It pairs:
- **Exact Mathematical Programming (HiGHS MILP)** for rigorous planning analysis and mathematical optimality proofs.
- **Submodular Greedy Heuristics** for sub-second interactive scenario exploration in web-based decision-support tools.

Tested against a real-world Florida corridor testbed of **2,413 candidate project parcels** and **1,200 Census demand centroids**.

---

## 2. Mathematical Formulation

Let:
$$x_j = \begin{cases} 1 & \text{facility selected at candidate } j \\ 0 & \text{otherwise} \end{cases}$$

$$y_i = \begin{cases} 1 & \text{demand node } i \text{ is covered} \\ 0 & \text{otherwise} \end{cases}$$

### Objective Function
Maximize normalized system efficiency and normalized disadvantaged community coverage:

$$\max_{x, y} \quad Z = (1 - w_{\text{eq}}) \frac{\sum_{i \in \mathcal{I}} D_i y_i}{\sum_{i \in \mathcal{I}} D_i} + w_{\text{eq}} \frac{\sum_{i \in \mathcal{I}_{\text{DAC}}} D_i y_i}{\sum_{i \in \mathcal{I}_{\text{DAC}}} D_i} - \lambda \sum_{j \in \mathcal{J}} c_j x_j$$

### Constraints
$$\begin{aligned}
y_i &\le \sum_{j \in \mathcal{N}_i} x_j && \forall i \in \mathcal{I} && \text{(Coverage Coupling)} \\
\sum_{j \in \mathcal{J}} c_j x_j &\le B && && \text{(Capital Budget Constraint)} \\
\sum_{j \in \mathcal{J}} x_j &\le P && && \text{(Facility Cardinality Cap)} \\
x_j + x_k &\le 1 && \forall (j, k) \in \mathcal{C} && \text{(Anti-Cannibalization Spacing)} \\
\sum_{i \in \mathcal{I}_{\text{DAC}}} D_i y_i &\ge \alpha \sum_{i \in \mathcal{I}} D_i y_i && && \text{(Linear Equity-Share Constraint)} \\
x_j &\in \{0, 1\}, \quad y_i \in [0, 1] && && \text{(Integrality \& Domain Bounds)}
\end{aligned}$$

---

## 3. Solver Benchmark: HiGHS MILP vs Greedy Marginal-Gain Heuristic

| Metric / Dimension | HiGHS MILP | Greedy Marginal-Gain Heuristic |
| :--- | :--- | :--- |
| **Algorithm** | Dual Simplex + Branch-and-Cut | Incremental Benefit-to-Cost Ratio |
| **Corridor Subset (500 Sites)** | **28 ms** (Proven optimal, solver gap = 0.0%) | **25 ms** (Fast heuristic) |
| **Full Network (2,413 Sites)** | ~6.0 sec (Feasible incumbent, gap reported) | **~0.71 sec** (Sub-second interactive) |
| **Optimality Information** | Dual bound, best incumbent, solver MIP gap | Empirical gap relative to MILP |
| **Primary Role** | Rigorous planning benchmark | Sub-second interactive scenario exploration |

> **Note on Approximation Guarantees:** Classical submodular greedy guarantees ($\ge 1 - 1/e \approx 63.2\%$) strictly apply to monotone submodular maximization under uniform cardinality constraints ($\sum x_j \le P$) without knapsack costs or pairwise exclusions. Under non-uniform capital costs ($\sum c_j x_j \le B$), minimum pairwise spatial separation ($x_j + x_k \le 1$), and equity constraints, theoretical guarantees become complex. This framework directly benchmarks heuristic performance using **empirical optimality gaps** measured against HiGHS dual bounds ($Gap = \frac{Z_{\text{MILP}} - Z_{\text{Greedy}}}{Z_{\text{MILP}}} \times 100\%$), which consistently remain within $2.0\% - 5.5\%$.

---

## 4. Sector-Specific Planning Models

The optimization engine is reusable across sectors, but each application incorporates distinct demand drivers, cost ranges, and planning priorities:

| Sector | Demand Metric ($D_i$) | Candidate Pool | Planning Cost Assumptions ($C_j$) | Equity Definition ($\mathcal{I}_{\text{DAC}}$) | Planning Reference |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **⚡ EV Charging** | Observed AADT + Commercial POI Density | Highway Exits & Retail Hubs | \$160k – \$320k | Disadvantaged communities with high transit/energy burden | NEVI Formula Program Standards |
| **🚌 Transit Stops** | Zero-Vehicle Households + Job Density | Intersections & Activity Hubs | \$45k – \$130k | Title VI protected areas & low-income Census tracts | FTA Title VI & Capital Grants |
| **🛡️ Safety** | 5-Year Fatal & Serious Injury (FSI) Crashes | High Injury Network Intersections | \$75k – \$240k | Vulnerable road users in underserved corridors | Safe Streets and Roads for All (SS4A) |
| **🚲 Bike Infrastructure** | Level of Traffic Stress (LTS) Gap + Commute Volume | Street Segments & Trail Links | \$65k – \$175k | Low-income car-free commute sheds | Reconnecting Communities / Active Transportation |

---

## 5. Data Provenance & Reproducibility

All empirical numbers in this repository are reproducible through real Florida open data:

### Primary Open Data Sources
| Dataset | Source Agency | Description & Timeframe |
| :--- | :--- | :--- |
| **Roadway & Traffic** | Florida Department of Transportation (FDOT) | Roadway Characteristics Inventory & 2023 Observed AADT |
| **Demographics** | US Census Bureau | American Community Survey (ACS) 5-Year Estimates (2018–2022) |
| **Crash Records** | Signal Four Analytics / FDOT CAR | 5-Year Fatal & Serious Injury (FSI) Crashes (2019–2023) |
| **Equity Designation** | Federal & State Screening Criteria | Census tracts with poverty rate > 18% or zero-car household share > 11.5% |

### Derived Optimization Entities
| Entity | Scale | Methodology |
| :--- | :--- | :--- |
| **Derived Candidate Sites** | 2,413 locations | Parcels synthesized from highway exits, transit stops, arterial intersections, and commercial hubs |
| **Derived Demand Centroids** | 1,200 zones | Population and latent travel demand centroids derived from US Census block groups |
| **Geography** | Central Florida Corridor | Orlando & Tampa Bay metropolitan commuting shed |

---

## 6. Quickstart & Usage

### 6.1 Installation
```bash
git clone git@github.com:dy-chang/transport-infra-optimization-lab.git
cd transport-infra-optimization-lab

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install in editable mode
pip install -e .
```

### 6.2 Running the Multi-Sector Pareto Analysis
```bash
python examples/run_pareto_analysis.py
```

### 6.3 Launching the Interactive Web Dashboard
```bash
streamlit run app.py
```
- Move sliders for **Budget** (`$2M ──●── $25M`), **Equity Weight** (`Low ──●── High`), and **Min Equity Share** (`0% ──●── 50%`).
- Optimal selected sites update interactively on the 3D PyDeck map.
- View Pareto Frontier trade-off curves, Sensitivity Sweeps, and export the capital improvement plan as a CSV.

### 6.4 Running Automated Verification Tests
```bash
pytest tests/ -v
```
All 10 unit/integration tests validate budget compliance, cardinality bounds, minimum spacing, linear equity-share constraints, sensitivity monotonicity, and the MILP $\ge$ Greedy optimality verification.

---

## 7. License
Distributed under the MIT License. See `LICENSE` for details.
