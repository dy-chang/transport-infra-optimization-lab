"""
Solution evaluation, optimality gap computation, and equity compliance analytics.
"""

from typing import Any, Dict, List
import numpy as np

from src.models import OptimizationSolution, ParetoPoint


def compute_optimality_gap(mip_sol: OptimizationSolution, greedy_sol: OptimizationSolution) -> Dict[str, Any]:
    """
    Compute relative optimality gap and computational speedup between exact MIP and Greedy heuristic.
    """
    z_mip = mip_sol.objective_value
    z_grd = greedy_sol.objective_value

    if abs(z_mip) > 1e-6:
        gap_pct = max(0.0, ((z_mip - z_grd) / z_mip) * 100.0)
    else:
        gap_pct = 0.0

    speedup = mip_sol.solve_time_sec / max(greedy_sol.solve_time_sec, 1e-4)

    return {
        "mip_objective": round(z_mip, 4),
        "greedy_objective": round(z_grd, 4),
        "optimality_gap_pct": round(gap_pct, 2),
        "mip_runtime_sec": round(mip_sol.solve_time_sec, 3),
        "greedy_runtime_sec": round(greedy_sol.solve_time_sec, 4),
        "speedup_factor": round(speedup, 1),
        "solution_agreement_pct": round(
            len(set(mip_sol.selected_site_ids) & set(greedy_sol.selected_site_ids))
            / max(len(mip_sol.selected_site_ids), 1) * 100.0, 1
        ),
    }


def evaluate_equity_fairness(solution: OptimizationSolution) -> Dict[str, Any]:
    """
    Evaluate equity share threshold compliance and distribution.
    """
    target_share = solution.scenario.min_equity_share * 100.0
    actual_share = solution.dac_share_pct
    delta = actual_share - target_share

    status = "COMPLIANT" if delta >= -0.5 else "DEFICIT"

    return {
        "target_share_pct": round(target_share, 1),
        "actual_dac_share_pct": round(actual_share, 1),
        "compliance_status": status,
        "share_margin_pct": round(delta, 1),
        "dac_demand_covered": round(solution.dac_demand_covered, 1),
        "total_demand_covered": round(solution.total_demand_covered, 1),
    }
