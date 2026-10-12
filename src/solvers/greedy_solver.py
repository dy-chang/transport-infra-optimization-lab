"""
High-performance greedy marginal-gain heuristic.
Executes in median <30ms on corridor subsets for sub-second interactive scenario exploration.
"""

import time
from typing import List, Optional
import numpy as np

from src.models import CandidateSite, DemandNode, OptimizationSolution, ScenarioConfig
from src.solvers.base import BaseSolver


class GreedyCoverageSolver(BaseSolver):
    """
    Greedy Marginal-Gain Heuristic Solver.
    Iteratively funds candidate facilities with the highest incremental normalized coverage benefit per dollar.
    """

    def __init__(self):
        super().__init__(name="Greedy Marginal-Gain Heuristic")

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

        cand_costs = np.array([c.cost for c in candidates], dtype=np.float64)
        dem_values = np.array([d.demand for d in demand_nodes], dtype=np.float64)
        is_dac = np.array([d.is_dac for d in demand_nodes], dtype=bool)

        total_demand_sum = float(np.sum(dem_values))
        dac_demand_sum = float(np.sum(dem_values[is_dac])) if np.any(is_dac) else 1.0

        coverage_mask = dist_cand_to_dem <= scenario.coverage_radius_miles

        cand_lats = np.array([c.latitude for c in candidates])
        cand_lons = np.array([c.longitude for c in candidates])

        selected_indices: List[int] = []
        covered_demand_mask = np.zeros(n_dem, dtype=bool)
        current_cost = 0.0

        w_eff = 1.0 - scenario.equity_weight
        w_eq = scenario.equity_weight

        # Precompute normalized value contributions per node
        node_eff_weights = (w_eff / total_demand_sum) * dem_values if total_demand_sum > 0 else np.zeros(n_dem)
        node_eq_weights = (w_eq / dac_demand_sum) * (dem_values * is_dac) if dac_demand_sum > 0 else np.zeros(n_dem)
        node_weights = node_eff_weights + node_eq_weights

        available_candidates = set(range(n_cand))

        for _ in range(scenario.max_facilities):
            best_cand = -1
            best_ratio = -1.0
            best_gain = 0.0

            for j in available_candidates:
                cost_j = cand_costs[j]
                if current_cost + cost_j > scenario.budget:
                    continue

                # Spacing constraint check
                if scenario.min_spacing_miles > 0 and len(selected_indices) > 0:
                    sel_lats = cand_lats[selected_indices]
                    sel_lons = cand_lons[selected_indices]
                    dlat = np.abs(sel_lats - cand_lats[j]) * 69.0
                    dlon = np.abs(sel_lons - cand_lons[j]) * 55.0
                    dists = np.sqrt(dlat**2 + dlon**2)
                    if np.any(dists < scenario.min_spacing_miles):
                        continue

                newly_covered = coverage_mask[:, j] & (~covered_demand_mask)
                marginal_gain = np.sum(node_weights[newly_covered])

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

            selected_indices.append(best_cand)
            available_candidates.remove(best_cand)
            current_cost += cand_costs[best_cand]
            covered_demand_mask |= coverage_mask[:, best_cand]

        solve_time = time.time() - t0

        selected_sites = [candidates[idx].site_id for idx in selected_indices]
        covered_nodes = [demand_nodes[i].node_id for i in np.where(covered_demand_mask)[0]]

        total_demand_covered = float(np.sum(dem_values[covered_demand_mask]))
        pct_demand_covered = (total_demand_covered / total_demand_sum * 100.0) if total_demand_sum > 0 else 0.0

        dac_demand_covered = float(np.sum(dem_values[covered_demand_mask & is_dac]))
        dac_share_pct = (dac_demand_covered / total_demand_covered * 100.0) if total_demand_covered > 0 else 0.0

        is_share_met = (dac_share_pct >= (scenario.min_equity_share * 100.0 - 0.5))
        budget_util_pct = (current_cost / scenario.budget * 100.0) if scenario.budget > 0 else 0.0

        norm_eff = total_demand_covered / total_demand_sum if total_demand_sum > 0 else 0.0
        norm_eq = dac_demand_covered / dac_demand_sum if dac_demand_sum > 0 else 0.0
        obj_val = float(w_eff * norm_eff + w_eq * norm_eq)

        return OptimizationSolution(
            scenario=scenario,
            selected_site_ids=selected_sites,
            total_cost=round(current_cost, 2),
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
            covered_node_ids=covered_nodes,
        )
