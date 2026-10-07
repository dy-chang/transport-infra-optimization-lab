"""
High-performance submodular greedy marginal coverage heuristic.
Executes in <50ms for instantaneous 60fps web slider interactivity.
"""

import time
from typing import List, Optional, Set
import numpy as np

from src.models import CandidateSite, DemandNode, OptimizationSolution, ScenarioConfig
from src.solvers.base import BaseSolver


class GreedyCoverageSolver(BaseSolver):
    """
    Submodular Greedy Marginal Return Solver with Dynamic Equity Multipliers.
    Selects candidate facilities iteratively based on maximum incremental weighted demand per dollar.
    """

    def __init__(self):
        super().__init__(name="Submodular Greedy Heuristic")

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

        cand_costs = np.array([c.cost for c in candidates])
        dem_values = np.array([d.demand for d in demand_nodes])
        is_dac = np.array([d.is_dac for d in demand_nodes])

        # Binary coverage adjacency matrix: coverage_mask[i, j] is True if site j covers demand i
        # dist_cand_to_dem has shape (n_dem, n_cand)
        coverage_mask = dist_cand_to_dem <= scenario.coverage_radius_miles

        # Distance matrix for spacing check
        cand_lats = np.array([c.latitude for c in candidates])
        cand_lons = np.array([c.longitude for c in candidates])

        selected_indices: List[int] = []
        covered_demand_mask = np.zeros(n_dem, dtype=bool)
        current_cost = 0.0

        # Weighted demand vector
        w_eff = 1.0 - scenario.equity_weight
        w_eq = scenario.equity_weight
        # Equity multiplier (boost DAC node weights by 2.5x under high equity weight)
        weights = w_eff * dem_values + (w_eq * 2.5 * dem_values * is_dac)

        available_candidates = set(range(n_cand))

        for _ in range(scenario.max_facilities):
            best_cand = -1
            best_ratio = -1.0
            best_gain = 0.0

            for j in available_candidates:
                cost_j = cand_costs[j]
                if current_cost + cost_j > scenario.budget:
                    continue

                # Anti-cannibalization spacing check
                if scenario.min_spacing_miles > 0 and len(selected_indices) > 0:
                    sel_lats = cand_lats[selected_indices]
                    sel_lons = cand_lons[selected_indices]
                    # Compute approximate distance to selected
                    dlat = np.abs(sel_lats - cand_lats[j]) * 69.0
                    dlon = np.abs(sel_lons - cand_lons[j]) * 55.0
                    dists = np.sqrt(dlat**2 + dlon**2)
                    if np.any(dists < scenario.min_spacing_miles):
                        continue

                # Incremental newly covered demand
                newly_covered = coverage_mask[:, j] & (~covered_demand_mask)
                marginal_gain = np.sum(weights[newly_covered])

                if marginal_gain <= 0:
                    continue

                # Benefit-to-cost ratio
                ratio = marginal_gain / (cost_j ** 0.85)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_cand = j
                    best_gain = marginal_gain

            if best_cand == -1:
                break

            # Add best candidate
            selected_indices.append(best_cand)
            available_candidates.remove(best_cand)
            current_cost += cand_costs[best_cand]
            covered_demand_mask |= coverage_mask[:, best_cand]

        solve_time = time.time() - t0

        # Compute solution statistics
        selected_sites = [candidates[idx].site_id for idx in selected_indices]
        covered_nodes = [demand_nodes[i].node_id for i in np.where(covered_demand_mask)[0]]

        total_demand_covered = float(np.sum(dem_values[covered_demand_mask]))
        total_possible_demand = float(np.sum(dem_values))
        pct_demand_covered = (total_demand_covered / total_possible_demand * 100.0) if total_possible_demand > 0 else 0.0

        dac_demand_covered = float(np.sum(dem_values[covered_demand_mask & is_dac]))
        dac_share_pct = (dac_demand_covered / total_demand_covered * 100.0) if total_demand_covered > 0 else 0.0

        is_quota_met = (dac_share_pct >= (scenario.min_equity_quota * 100.0))
        budget_util_pct = (current_cost / scenario.budget * 100.0) if scenario.budget > 0 else 0.0

        obj_val = float(np.sum(weights[covered_demand_mask]) - (1e-6 * current_cost))

        return OptimizationSolution(
            scenario=scenario,
            selected_site_ids=selected_sites,
            total_cost=round(current_cost, 2),
            budget_utilization_pct=round(budget_util_pct, 1),
            total_demand_covered=round(total_demand_covered, 1),
            pct_total_demand_covered=round(pct_demand_covered, 1),
            dac_demand_covered=round(dac_demand_covered, 1),
            dac_share_pct=round(dac_share_pct, 1),
            is_equity_quota_met=is_quota_met,
            solver_name=self.name,
            solve_time_sec=round(solve_time, 4),
            objective_value=round(obj_val, 2),
            covered_node_ids=covered_nodes,
        )
