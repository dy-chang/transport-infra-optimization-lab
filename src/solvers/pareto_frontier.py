"""
Multi-objective Pareto Frontier analyzer.
Traces the trade-off curve between system efficiency (total demand) and equity (DAC coverage).
"""

from typing import List, Optional
import numpy as np

from src.models import CandidateSite, DemandNode, ParetoPoint, ScenarioConfig
from src.solvers.greedy_solver import GreedyCoverageSolver


class ParetoFrontierAnalyzer:
    """
    Computes empirical Pareto Trade-Off frontiers across equity weight sweeps.
    """

    def __init__(self):
        self.solver = GreedyCoverageSolver()

    def generate_frontier(
        self,
        base_scenario: ScenarioConfig,
        candidates: List[CandidateSite],
        demand_nodes: List[DemandNode],
        dist_cand_to_dem: np.ndarray,
        dist_cand_to_cand: Optional[np.ndarray] = None,
        weights: Optional[List[float]] = None,
    ) -> List[ParetoPoint]:
        if weights is None:
            weights = [0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.0]

        frontier: List[ParetoPoint] = []

        for w in weights:
            # Clone scenario with modified equity weight
            scenario = ScenarioConfig(
                domain_id=base_scenario.domain_id,
                budget=base_scenario.budget,
                max_facilities=base_scenario.max_facilities,
                equity_weight=float(w),
                min_equity_quota=0.0,  # Unconstrain quota during frontier generation to observe pure trade-off
                coverage_radius_miles=base_scenario.coverage_radius_miles,
                min_spacing_miles=base_scenario.min_spacing_miles,
                solver_type="greedy",
            )

            sol = self.solver.solve(scenario, candidates, demand_nodes, dist_cand_to_dem, dist_cand_to_cand)
            frontier.append(
                ParetoPoint(
                    equity_weight=float(w),
                    efficiency_score=sol.total_demand_covered,
                    equity_score=sol.dac_demand_covered,
                    total_cost=sol.total_cost,
                    selected_count=len(sol.selected_site_ids),
                    dac_share_pct=sol.dac_share_pct,
                )
            )

        return frontier
