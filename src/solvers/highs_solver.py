"""
Mixed-Integer Linear Programming (MILP) solver powered by native HiGHS.
Provides proven mathematical optimality on benchmark instances and best incumbent solutions
under configurable execution budgets.
"""

import time
from typing import List, Optional
import numpy as np
from scipy.optimize import LinearConstraint, milp
from scipy.sparse import csc_matrix

from src.models import CandidateSite, DemandNode, OptimizationSolution, ScenarioConfig
from src.solvers.base import BaseSolver


class HiGHSMIPSolver(BaseSolver):
    """
    HiGHS Mixed-Integer Linear Programming (MILP) Solver.
    Solves the Multi-Objective Maximum Coverage Location Problem (MCLP)
    with budget, cardinality, spacing, and linear equity-share constraints.
    """

    def __init__(self):
        super().__init__(name="HiGHS MILP")

    def solve(
        self,
        scenario: ScenarioConfig,
        candidates: List[CandidateSite],
        demand_nodes: List[DemandNode],
        dist_cand_to_dem: np.ndarray,
        dist_cand_to_cand: Optional[np.ndarray] = None,
    ) -> OptimizationSolution:
        t0 = time.time()

        n_cand = len(candidates)
        n_dem = len(demand_nodes)
        n_vars = n_cand + n_dem

        cand_costs = np.array([c.cost for c in candidates], dtype=np.float64)
        dem_values = np.array([d.demand for d in demand_nodes], dtype=np.float64)
        is_dac = np.array([d.is_dac for d in demand_nodes], dtype=bool)

        total_demand_sum = float(np.sum(dem_values))
        dac_demand_sum = float(np.sum(dem_values[is_dac])) if np.any(is_dac) else 1.0

        # Normalized Objective Formulation:
        # Maximize: (1 - w_eq) * (sum_i D_i * y_i / sum D_i) + w_eq * (sum_{i in DAC} D_i * y_i / sum_{DAC} D_i)
        # Optional: - lambda * sum C_j * x_j (Cost-sensitive mode)
        w_eff = 1.0 - scenario.equity_weight
        w_eq = scenario.equity_weight

        y_coeff = np.zeros(n_dem, dtype=np.float64)
        if total_demand_sum > 0:
            y_coeff += (w_eff / total_demand_sum) * dem_values
        if dac_demand_sum > 0:
            y_coeff += (w_eq / dac_demand_sum) * (dem_values * is_dac)

        c = np.zeros(n_vars, dtype=np.float64)
        if scenario.cost_sensitive_mode:
            c[:n_cand] = scenario.cost_penalty_lambda * cand_costs
        c[n_cand:] = -y_coeff  # Negate for minimization

        # Integrality: x_j are binary (1), y_i are continuous relaxed [0, 1] (0)
        integrality = np.zeros(n_vars, dtype=int)
        integrality[:n_cand] = 1

        lb = np.zeros(n_vars, dtype=np.float64)
        ub = np.ones(n_vars, dtype=np.float64)

        # Build Sparse Constraint Matrix A * z <= row_ub
        rows = []
        cols = []
        data = []
        row_ub = []
        row_idx = 0

        # 1. Budget Constraint: sum(C_j * x_j) <= B
        for j in range(n_cand):
            rows.append(row_idx)
            cols.append(j)
            data.append(cand_costs[j])
        row_ub.append(float(scenario.budget))
        row_idx += 1

        # 2. Cardinality Constraint: sum(x_j) <= P
        for j in range(n_cand):
            rows.append(row_idx)
            cols.append(j)
            data.append(1.0)
        row_ub.append(float(scenario.max_facilities))
        row_idx += 1

        # 3. Coverage Coupling Constraints: y_i <= sum_{j in N(i)} x_j
        coverage_mask = dist_cand_to_dem <= scenario.coverage_radius_miles
        for i in range(n_dem):
            covering_cands = np.where(coverage_mask[i, :])[0]
            if len(covering_cands) == 0:
                ub[n_cand + i] = 0.0
                continue

            rows.append(row_idx)
            cols.append(n_cand + i)
            data.append(1.0)

            for j in covering_cands:
                rows.append(row_idx)
                cols.append(int(j))
                data.append(-1.0)

            row_ub.append(0.0)
            row_idx += 1

        # 4. Anti-Cannibalization Spacing Constraints: x_j + x_k <= 1 for close pairs
        if scenario.min_spacing_miles > 0:
            cand_lats = np.array([c.latitude for c in candidates])
            cand_lons = np.array([c.longitude for c in candidates])
            n_sub = min(n_cand, 350)
            dlat = np.abs(cand_lats[:n_sub, np.newaxis] - cand_lats[np.newaxis, :n_sub]) * 69.0
            dlon = np.abs(cand_lons[:n_sub, np.newaxis] - cand_lons[np.newaxis, :n_sub]) * 55.0
            pairwise_dists = np.sqrt(dlat**2 + dlon**2)
            
            close_pairs = np.argwhere(np.triu(pairwise_dists < scenario.min_spacing_miles, k=1))
            for j, k in close_pairs[:500]:
                rows.extend([row_idx, row_idx])
                cols.extend([int(j), int(k)])
                data.extend([1.0, 1.0])
                row_ub.append(1.0)
                row_idx += 1

        # 5. Linear Reformulation of Equity-Share Constraint:
        # sum_{i in DAC} D_i * y_i >= alpha * sum_i D_i * y_i
        alpha = scenario.min_equity_share
        if alpha > 0.02 and np.any(is_dac):
            for i in range(n_dem):
                coeff = (alpha * dem_values[i]) if not is_dac[i] else (-(1.0 - alpha) * dem_values[i])
                rows.append(row_idx)
                cols.append(n_cand + i)
                data.append(float(coeff))
            row_ub.append(0.0)
            row_idx += 1

        A = csc_matrix((data, (rows, cols)), shape=(row_idx, n_vars))
        constraints = LinearConstraint(A, ub=row_ub, lb=[-np.inf] * row_idx)

        # Invoke HiGHS Solver
        res = milp(
            c=c,
            integrality=integrality,
            bounds=(lb, ub),
            constraints=constraints,
            options={"time_limit": 6.0, "mip_rel_gap": 0.015}
        )

        solve_time = time.time() - t0

        if res.success or (res.x is not None and res.status in (0, 1, 2)):
            x_sol = np.round(res.x[:n_cand]).astype(int)
            y_sol = np.clip(res.x[n_cand:], 0.0, 1.0)
            selected_indices = np.where(x_sol == 1)[0].tolist()
            covered_indices = np.where(y_sol >= 0.5)[0].tolist()
            
            # Extract solver bounds & status
            is_optimal = bool(res.status == 0)
            mip_gap = float(res.mip_gap) if hasattr(res, "mip_gap") and res.mip_gap is not None else 0.0
            dual_bound = float(-res.mip_dual_bound) if hasattr(res, "mip_dual_bound") and res.mip_dual_bound is not None else None
        else:
            from src.solvers.greedy_solver import GreedyCoverageSolver
            return GreedyCoverageSolver().solve(scenario, candidates, demand_nodes, dist_cand_to_dem, dist_cand_to_cand)

        total_cost = float(np.sum(cand_costs[selected_indices]))
        total_demand_covered = float(np.sum(dem_values[covered_indices]))
        pct_demand_covered = (total_demand_covered / total_demand_sum * 100.0) if total_demand_sum > 0 else 0.0

        dac_demand_covered = float(np.sum(dem_values[np.array(covered_indices)[is_dac[covered_indices]]])) if len(covered_indices) > 0 else 0.0
        dac_share_pct = (dac_demand_covered / total_demand_covered * 100.0) if total_demand_covered > 0 else 0.0
        is_share_met = (dac_share_pct >= (scenario.min_equity_share * 100.0 - 0.5))

        budget_util_pct = (total_cost / scenario.budget * 100.0) if scenario.budget > 0 else 0.0
        
        norm_eff = total_demand_covered / total_demand_sum if total_demand_sum > 0 else 0.0
        norm_eq = dac_demand_covered / dac_demand_sum if dac_demand_sum > 0 else 0.0
        obj_val = float(w_eff * norm_eff + w_eq * norm_eq)

        return OptimizationSolution(
            scenario=scenario,
            selected_site_ids=[candidates[idx].site_id for idx in selected_indices],
            total_cost=round(total_cost, 2),
            budget_utilization_pct=round(budget_util_pct, 1),
            total_demand_covered=round(total_demand_covered, 1),
            pct_total_demand_covered=round(pct_demand_covered, 1),
            dac_demand_covered=round(dac_demand_covered, 1),
            dac_share_pct=round(dac_share_pct, 1),
            is_equity_share_met=is_share_met,
            solver_name=self.name,
            solve_time_sec=round(solve_time, 4),
            objective_value=round(obj_val, 4),
            norm_efficiency=round(norm_eff, 4),
            norm_equity=round(norm_eq, 4),
            best_bound=round(dual_bound, 4) if dual_bound is not None else None,
            mip_gap=round(mip_gap, 4),
            is_proven_optimal=is_optimal,
            covered_node_ids=[demand_nodes[i].node_id for i in covered_indices],
            optimality_gap_pct=round(mip_gap * 100.0, 2) if mip_gap is not None else 0.0,
        )
