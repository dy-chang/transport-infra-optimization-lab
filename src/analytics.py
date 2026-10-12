"""
Solution evaluation, optimality gap computation, and equity compliance analytics.
"""

from typing import Any, Dict, List, Optional
import numpy as np

from src.models import OptimizationSolution, ParetoPoint, ScenarioConfig


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


def run_sensitivity_analysis(
    scenario: ScenarioConfig,
    candidates: List[Any],
    demand_nodes: List[Any],
    dist_cand_to_dem: np.ndarray,
    dist_cand_to_cand: Optional[np.ndarray] = None,
    parameter_name: str = "budget",
    parameter_values: Optional[List[Any]] = None,
    solver_type: str = "greedy",
) -> List[Any]:
    """
    Parametric sensitivity analysis sweeping a specified decision or policy parameter.
    Evaluates system performance trade-offs across budget levels, equity weights, coverage radii, etc.
    """
    from src.models import SensitivityResult

    if parameter_values is None:
        if parameter_name == "budget":
            base_b = scenario.budget
            parameter_values = [base_b * factor for factor in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]]
        elif parameter_name == "equity_weight":
            parameter_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        elif parameter_name == "coverage_radius_miles":
            parameter_values = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
        elif parameter_name == "max_facilities":
            base_m = scenario.max_facilities
            parameter_values = sorted(list({max(1, int(base_m * f)) for f in [0.2, 0.5, 0.8, 1.0, 1.5, 2.0]}))
        elif parameter_name == "min_equity_share":
            parameter_values = [0.1, 0.2, 0.3, 0.4, 0.5]
        else:
            raise ValueError(f"Unknown parameter_name: {parameter_name}")

    if solver_type == "highs":
        from src.solvers.highs_solver import HiGHSMIPSolver
        solver = HiGHSMIPSolver()
    else:
        from src.solvers.greedy_solver import GreedyCoverageSolver
        solver = GreedyCoverageSolver()

    results: List[SensitivityResult] = []

    for val in parameter_values:
        scenario_dict = scenario.model_dump()
        scenario_dict[parameter_name] = val
        scenario_dict["solver_type"] = solver_type
        mod_scenario = ScenarioConfig(**scenario_dict)

        sol = solver.solve(
            mod_scenario,
            candidates,
            demand_nodes,
            dist_cand_to_dem,
            dist_cand_to_cand,
        )

        results.append(
            SensitivityResult(
                parameter_name=parameter_name,
                parameter_value=val,
                total_demand_covered=sol.total_demand_covered,
                dac_demand_covered=sol.dac_demand_covered,
                dac_share_pct=sol.dac_share_pct,
                total_cost=sol.total_cost,
                facility_count=len(sol.selected_site_ids),
            )
        )

    return results
