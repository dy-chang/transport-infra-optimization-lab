"""
Extended test suite for Transportation Infrastructure Optimization Lab.
Validates:
1. Budget constraint satisfaction (sum c_j * x_j <= B)
2. Facility cardinality constraint satisfaction (sum x_j <= P)
3. Minimum spacing constraint satisfaction (no conflicting pairs selected)
4. Equity-share constraint satisfaction (DAC share >= alpha)
5. Coverage logic correctness (node covered iff dist <= R from funded site)
6. Theoretical optimality guarantee: MILP objective >= Greedy objective
7. Infeasible / tight scenario handling
8. Deterministic reproducibility across repeated runs
9. Sector-specific planning models (EV, Transit, Safety, Bike)
"""

import pytest
import numpy as np

from src.config import SECTOR_PROFILES
from src.data_loader import InfrastructureDataLoader, haversine_distance_matrix
from src.models import ScenarioConfig
from src.solvers.greedy_solver import GreedyCoverageSolver
from src.solvers.highs_solver import HiGHSMIPSolver
from src.solvers.pareto_frontier import ParetoFrontierAnalyzer
from src.analytics import compute_optimality_gap, evaluate_equity_fairness


@pytest.fixture(scope="session")
def test_dataset():
    """Session fixture generating a fast calibrated testbed."""
    loader = InfrastructureDataLoader(seed=123)
    candidates, demand_nodes, dist_cand_to_dem, dist_cand_to_cand = loader.generate_testbed(
        sector_id="EV_CHARGING",
        n_candidates=350,
        n_demand_nodes=220,
    )
    return candidates, demand_nodes, dist_cand_to_dem, dist_cand_to_cand


def test_budget_constraint_satisfied(test_dataset):
    candidates, demand_nodes, dist_cand_to_dem, dist_cand_to_cand = test_dataset
    budget = 1_800_000.0
    scenario = ScenarioConfig(
        sector_id="EV_CHARGING",
        budget=budget,
        max_facilities=15,
        coverage_radius_miles=3.0,
        solver_type="greedy",
    )
    sol = GreedyCoverageSolver().solve(scenario, candidates, demand_nodes, dist_cand_to_dem, dist_cand_to_cand)
    assert sol.total_cost <= budget, f"Cost {sol.total_cost} exceeded budget {budget}"


def test_facility_count_constraint_satisfied(test_dataset):
    candidates, demand_nodes, dist_cand_to_dem, dist_cand_to_cand = test_dataset
    max_fac = 6
    scenario = ScenarioConfig(
        sector_id="EV_CHARGING",
        budget=10_000_000.0,
        max_facilities=max_fac,
        coverage_radius_miles=3.0,
        solver_type="greedy",
    )
    sol = GreedyCoverageSolver().solve(scenario, candidates, demand_nodes, dist_cand_to_dem, dist_cand_to_cand)
    assert len(sol.selected_site_ids) <= max_fac, f"Selected {len(sol.selected_site_ids)} > max {max_fac}"


def test_minimum_spacing_constraint_satisfied(test_dataset):
    candidates, demand_nodes, dist_cand_to_dem, dist_cand_to_cand = test_dataset
    min_dist = 1.2  # miles
    scenario = ScenarioConfig(
        sector_id="EV_CHARGING",
        budget=5_000_000.0,
        max_facilities=12,
        coverage_radius_miles=3.0,
        min_spacing_miles=min_dist,
        solver_type="greedy",
    )
    sol = GreedyCoverageSolver().solve(scenario, candidates, demand_nodes, dist_cand_to_dem, dist_cand_to_cand)
    
    sel_sites = [c for c in candidates if c.site_id in sol.selected_site_ids]
    if len(sel_sites) > 1:
        lats = np.array([c.latitude for c in sel_sites])
        lons = np.array([c.longitude for c in sel_sites])
        pair_dists = haversine_distance_matrix(lats, lons, lats, lons)
        np.fill_diagonal(pair_dists, np.inf)
        min_observed = pair_dists.min()
        assert min_observed >= (min_dist - 0.05), f"Found sites spaced {min_observed:.3f} < {min_dist} miles"


def test_equity_share_constraint_satisfied(test_dataset):
    candidates, demand_nodes, dist_cand_to_dem, dist_cand_to_cand = test_dataset
    target_share = 0.35  # 35% minimum DAC share
    scenario = ScenarioConfig(
        sector_id="EV_CHARGING",
        budget=3_000_000.0,
        max_facilities=10,
        equity_weight=0.6,
        min_equity_share=target_share,
        coverage_radius_miles=3.0,
        solver_type="highs",
    )
    sol = HiGHSMIPSolver().solve(scenario, candidates, demand_nodes, dist_cand_to_dem, dist_cand_to_cand)
    assert sol.dac_share_pct >= (target_share * 100.0 - 0.5)


def test_coverage_logic_correct(test_dataset):
    candidates, demand_nodes, dist_cand_to_dem, dist_cand_to_cand = test_dataset
    radius = 2.5
    scenario = ScenarioConfig(
        sector_id="EV_CHARGING",
        budget=2_000_000.0,
        max_facilities=6,
        coverage_radius_miles=radius,
        solver_type="greedy",
    )
    sol = GreedyCoverageSolver().solve(scenario, candidates, demand_nodes, dist_cand_to_dem, dist_cand_to_cand)

    sel_idx = [i for i, c in enumerate(candidates) if c.site_id in sol.selected_site_ids]
    for n_id in sol.covered_node_ids:
        dem_idx = next(i for i, d in enumerate(demand_nodes) if d.node_id == n_id)
        min_d = dist_cand_to_dem[dem_idx, sel_idx].min()
        assert min_d <= radius + 1e-4, f"Node {n_id} marked covered but closest facility is {min_d:.3f} mi > {radius} mi"


def test_milp_objective_ge_greedy_objective(test_dataset):
    """Core Operations Research verification: Exact MILP objective must be >= Heuristic objective."""
    candidates, demand_nodes, dist_cand_to_dem, dist_cand_to_cand = test_dataset
    scenario = ScenarioConfig(
        sector_id="EV_CHARGING",
        budget=1_500_000.0,
        max_facilities=6,
        equity_weight=0.35,
        min_equity_share=0.10,
        coverage_radius_miles=3.0,
        min_spacing_miles=0.5,
        solver_type="highs",
    )

    mip_sol = HiGHSMIPSolver().solve(scenario, candidates, demand_nodes, dist_cand_to_dem, dist_cand_to_cand)
    greedy_sol = GreedyCoverageSolver().solve(scenario, candidates, demand_nodes, dist_cand_to_dem, dist_cand_to_cand)

    # HiGHS global branch-and-cut optimum must be >= Greedy (allowing negligible 1e-3 numerical tolerance)
    assert mip_sol.objective_value >= greedy_sol.objective_value - 1e-3, (
        f"MILP objective ({mip_sol.objective_value}) was strictly lower than Greedy ({greedy_sol.objective_value})"
    )


def test_infeasible_tight_scenario_handling(test_dataset):
    candidates, demand_nodes, dist_cand_to_dem, dist_cand_to_cand = test_dataset
    # Budget $50,000 is far below minimum unit cost (~$160,000)
    tight_scenario = ScenarioConfig(
        sector_id="EV_CHARGING",
        budget=50_000.0,
        max_facilities=5,
        coverage_radius_miles=2.0,
        solver_type="greedy",
    )
    sol = GreedyCoverageSolver().solve(tight_scenario, candidates, demand_nodes, dist_cand_to_dem, dist_cand_to_cand)
    assert len(sol.selected_site_ids) == 0
    assert sol.total_cost == 0.0
    assert sol.total_demand_covered == 0.0


def test_deterministic_reproducible_results(test_dataset):
    candidates, demand_nodes, dist_cand_to_dem, dist_cand_to_cand = test_dataset
    scenario = ScenarioConfig(
        sector_id="EV_CHARGING",
        budget=2_500_000.0,
        max_facilities=8,
        equity_weight=0.40,
        coverage_radius_miles=2.5,
        solver_type="greedy",
    )
    solver = GreedyCoverageSolver()
    sol1 = solver.solve(scenario, candidates, demand_nodes, dist_cand_to_dem, dist_cand_to_cand)
    sol2 = solver.solve(scenario, candidates, demand_nodes, dist_cand_to_dem, dist_cand_to_cand)

    assert sol1.selected_site_ids == sol2.selected_site_ids
    assert sol1.objective_value == sol2.objective_value
    assert sol1.total_cost == sol2.total_cost


def test_sector_specific_profiles():
    """Verify distinct cost bounds, demand metrics, and attributes across all 4 sectors."""
    loader = InfrastructureDataLoader(seed=42)
    sectors = ["EV_CHARGING", "TRANSIT_STOPS", "SAFETY_VISION_ZERO", "BIKE_MICROMOBILITY"]

    for sec in sectors:
        profile = SECTOR_PROFILES[sec]
        cands, dems, dist_mat, _ = loader.generate_testbed(sector_id=sec, n_candidates=100, n_demand_nodes=80)
        
        costs = [c.cost for c in cands]
        assert min(costs) >= profile.unit_cost_min
        assert max(costs) <= profile.unit_cost_max
        assert cands[0].sector_id == sec

        scenario = ScenarioConfig(
            sector_id=sec,
            budget=profile.default_budget * 0.25,
            max_facilities=8,
            equity_weight=profile.default_equity_weight,
            coverage_radius_miles=profile.coverage_radius_miles,
            min_spacing_miles=profile.min_spacing_miles,
        )
        sol = GreedyCoverageSolver().solve(scenario, cands, dems, dist_mat)
        assert len(sol.selected_site_ids) > 0
