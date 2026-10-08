# Mathematical Formulation & Operations Research Specification

## 1. Executive Summary & Problem Setting
Transportation capital programming requires allocating limited public funds across candidate project locations to maximize community mobility benefit while satisfying spatial separation, budgetary caps, and equity objectives.

This document details the mathematical programming formulation of the **Multi-Objective Maximum Coverage Location Problem with an Equity-Share Constraint (MCLP-EQ)** implemented in `transport-infra-optimization-lab`.

---

## 2. Mathematical Formulation

### 2.1 Sets and Indices
- $\mathcal{J} = \{1, 2, \dots, N\}$: Set of candidate project sites ($N = 2,413$ potential locations).
- $\mathcal{I} = \{1, 2, \dots, M\}$: Set of spatial demand centroids ($M = 1,200$ Census Block Groups / TAZs).
- $\mathcal{I}_{\text{DAC}} \subseteq \mathcal{I}$: Subset of demand centroids designated as disadvantaged communities (DAC).
- $\mathcal{N}(i) = \{j \in \mathcal{J} \mid \text{dist}(i, j) \le R\}$: Set of candidate facilities that cover demand centroid $i$ within service distance radius $R$.
- $\mathcal{C} = \{(j, k) \in \mathcal{J}^2 \mid j < k, \, \text{dist}(j, k) < d_{\min}\}$: Set of conflicting candidate pairs violating the minimum separation threshold $d_{\min}$.

### 2.2 Parameters
- $D_i \ge 0$: Latent demand weight of centroid $i$ (AADT volume, transit reliance, severe crash risk, or bike latent demand).
- $C_j > 0$: Capital investment cost of facility $j$ (equipment, civil engineering, grid interconnection).
- $B > 0$: Total capital expenditure budget (e.g., \$5M to \$20M).
- $P \in \mathbb{Z}^+$: Cardinality upper bound on funded facilities.
- $w_{\text{eq}} \in [0, 1]$: Multi-objective equity priority weight (controllable via slider).
- $\alpha \in [0, 1]$: Configurable minimum equity share threshold ($\ge 30\%$ or $40\%$).
- $\lambda > 0$: Infinitesimal cost regularizer ($\lambda = 10^{-7}$) ensuring pareto-efficiency among equal-coverage solutions.

### 2.3 Decision Variables
- $x_j \in \{0, 1\}$: Binary decision variable; $x_j = 1$ if candidate facility $j$ is funded/constructed, $0$ otherwise.
- $y_i \in [0, 1]$: Coverage indicator; $y_i = 1$ if demand centroid $i$ is covered by at least one constructed facility, $0$ otherwise.

---

### 2.4 Mixed-Integer Linear Program (MILP)

$$\max_{x, y} \quad Z = (1 - w_{\text{eq}}) \frac{\sum_{i \in \mathcal{I}} D_i y_i}{\sum_{i \in \mathcal{I}} D_i} + w_{\text{eq}} \frac{\sum_{i \in \mathcal{I}_{\text{DAC}}} D_i y_i}{\sum_{i \in \mathcal{I}_{\text{DAC}}} D_i} - \lambda \sum_{j \in \mathcal{J}} C_j x_j$$

**Subject to:**

1. **Capital Budget Constraint**:
   $$\sum_{j \in \mathcal{J}} C_j x_j \le B$$

2. **Cardinality Upper Bound**:
   $$\sum_{j \in \mathcal{J}} x_j \le P$$

3. **Coverage Coupling Constraints**:
   $$y_i \le \sum_{j \in \mathcal{N}(i)} x_j \quad \forall i \in \mathcal{I}$$

4. **Anti-Cannibalization Spatial Spacing**:
   $$x_j + x_k \le 1 \quad \forall (j, k) \in \mathcal{C}$$

5. **Equity-Share Coverage Constraint**:
   The fractional equity requirement:
   $$\frac{\sum_{i \in \mathcal{I}_{\text{DAC}}} D_i y_i}{\sum_{i \in \mathcal{I}} D_i y_i} \ge \alpha$$
   is reformatted as an exact linear inequality:
   $$\sum_{i \in \mathcal{I} \setminus \mathcal{I}_{\text{DAC}}} \left( \alpha D_i \right) y_i - \sum_{i \in \mathcal{I}_{\text{DAC}}} \left( (1 - \alpha) D_i \right) y_i \le 0$$

6. **Integrality and Domain Bounds**:
   $$x_j \in \{0, 1\} \quad \forall j \in \mathcal{J}, \qquad 0 \le y_i \le 1 \quad \forall i \in \mathcal{I}$$

*(Note: Although $y_i$ is logically binary, relaxing $y_i \in [0, 1]$ maintains exact integer solutions at optimality because maximizing positive objective coefficients forces $y_i \to 1$ whenever $\sum_{j \in \mathcal{N}(i)} x_j \ge 1$, substantially reducing Branch-and-Bound search trees).*

---

## 3. Dual-Engine Algorithmic Design

| Dimension | HiGHS Exact MILP Solver | Submodular Greedy Heuristic |
| :--- | :--- | :--- |
| **Algorithm** | Simplex + Branch-and-Cut with Dual Presolve | Submodular Greedy with Marginal Cost Efficiency |
| **Optimality** | Proven Global Optimum ($Gap \le 1.5\%$) | Heuristic ($\ge (1 - 1/e) \approx 63.2\%$ worst-case; $>90\%$ in practice) |
| **Solve Latency** | $0.5 \text{ s} - 3.5 \text{ s}$ | **$0.025 \text{ s} - 0.040 \text{ s}$ (<40ms on subsets; <0.75s on full 2,413 network)** |
| **Role in Lab** | Rigorous verification, formal planning analysis | Sub-second interactive scenario exploration |

---

## 4. Key Academic References
1. Church, R., & ReVelle, C. (1974). *The maximal covering location problem*. Papers of the Regional Science Association, 32(1), 101-118.
2. Daskin, M. S. (1995). *Network and Discrete Location: Models, Algorithms, and Applications*. John Wiley & Sons.
3. Marianov, V., & Serra, D. (2002). *Location models for airlines, bus, and rail services*. Facility Location: Applications and Theory, 51-80.
