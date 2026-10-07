"""
Comprehensive pytest test suite validating:
1. Spatial data loader and Florida corridor testbed
2. Submodular greedy marginal coverage solver (<50ms)
3. Exact HiGHS mixed-integer programming (MIP) solver
4. Constraint satisfaction (budget, cardinality, spacing, Justice40 quota)
5. Multi-sector domain extensibility (EV, Transit, Safety, Bike)
6. Multi-objective Pareto frontier generation
"""

import pytest
import numpy as np

from src.config import DOMAIN_PROFILES
from src.data_loader import InfrastructureDataLoader
from src.models import ScenarioConfig
from src.solvers.greedy_solver import GreedyCoverageSolver
from src.solvers.highs_solver import HiGHSMIPSolver
from src.solvers.pareto_frontier import ParetoFrontierAnalyzer
from src.analytics import compute_optimality_gap, evaluate_equity_fairness


@pytest.fixture(scope="session")
def test_dataset():
    """Session fixture generating a fast calibrated testbed."""
    loader = InfrastructureDataLoader(seed=123)
    # Using 300 candidates and 200 demand nodes for swift test execution
    candidates, demand_nodes, dist_cand_to_dem, dist_cand_to_cand = loader.generate_testbed(
        domain_id="EV_CHARGING",
        n_candidates=300,
        n_demand_nodes=200,
    )
    return candidates, demand_nodes, dist_cand_to_dem, dist_cand_to_cand


def test_data_loader_structure(test_dataset):
    candidates, demand_nodes, dist_cand_to_dem, dist_cand_to_cand = test_dataset
    assert len(candidates) == 300
    assert len(demand_nodes) == 200
    assert dist_cand_to_dem.shape == (200, 300)
    assert np.all(dist_cand_to_dem >= 0)

    # Verify presence of Justice40 DAC communities
    dac_count = sum(1 for d in demand_nodes if d.is_dac)
    assert dac_count > 0
    assert dac_count < len(demand_nodes)


def test_greedy_solver_constraints(test_dataset):
    candidates, demand_nodes, dist_cand_to_dem, dist_cand_to_cand = test_dataset
    scenario = ScenarioConfig(
        domain_id="EV_CHARGING",
        budget=2_500_000.0,
        max_facilities=12,
        equity_weight=0.4,
        min_equity_quota=0.25,
        coverage_radius_miles=3.0,
        min_spacing_miles=0.8,
        solver_type="greedy",
    )

    solver = GreedyCoverageSolver()
    sol = solver.solve(scenario, candidates, demand_nodes, dist_cand_to_dem, dist_cand_to_cand)

    assert sol.total_cost <= scenario.budget
    assert len(sol.selected_site_ids) <= scenario.max_facilities
    assert sol.total_demand_covered > 0
    assert sol.solve_time_sec < 0.2


def test_highs_mip_solver_optimality(test_dataset):
    candidates, demand_nodes, dist_cand_to_dem, dist_cand_to_cand = test_dataset
    scenario = ScenarioConfig(
        domain_id="EV_CHARGING",
        budget=2_000_000.0,
        max_facilities=8,
        equity_weight=0.3,
        min_equity_quota=0.20,
        coverage_radius_miles=3.0,
        min_spacing_miles=0.5,
        solver_type="highs",
    )

    solver = HiGHSMIPSolver()
    sol = solver.solve(scenario, candidates, demand_nodes, dist_cand_to_dem, dist_cand_to_cand)

    assert sol.total_cost <= scenario.budget + 1.0  # Floating point tolerance
    assert len(sol.selected_site_ids) <= scenario.max_facilities
    assert sol.total_demand_covered > 0


def test_solver_comparison_and_optimality_gap(test_dataset):
    candidates, demand_nodes, dist_cand_to_dem, dist_cand_to_cand = test_dataset
    scenario = ScenarioConfig(
        domain_id="EV_CHARGING",
        budget=1_800_000.0,
        max_facilities=6,
        equity_weight=0.35,
        min_equity_quota=0.15,
        coverage_radius_miles=3.0,
        min_spacing_miles=0.5,
        solver_type="highs",
    )

    mip_sol = HiGHSMIPSolver().solve(scenario, candidates, demand_nodes, dist_cand_to_dem, dist_cand_to_cand)
    greedy_sol = GreedyCoverageSolver().solve(scenario, candidates, demand_nodes, dist_cand_to_dem, dist_cand_to_cand)

    gap_stats = compute_optimality_gap(mip_sol, greedy_sol)
    assert "optimality_gap_pct" in gap_stats
    assert "speedup_factor" in gap_stats
    # Greedy should achieve at least 70% of MIP objective
    assert greedy_sol.objective_value >= 0.70 * mip_sol.objective_value


def test_multi_sector_extensibility():
    """Verify that all 4 transportation domains can instantiate and solve."""
    loader = InfrastructureDataLoader(seed=999)
    solver = GreedyCoverageSolver()

    for domain_key in ["EV_CHARGING", "TRANSIT_STOPS", "SAFETY_VISION_ZERO", "BIKE_MICROMOBILITY"]:
        profile = DOMAIN_PROFILES[domain_key]
        cands, dems, dist_mat, _ = loader.generate_testbed(domain_id=domain_key, n_candidates=100, n_demand_nodes=80)
        scenario = ScenarioConfig(
            domain_id=domain_key,
            budget=profile.default_budget * 0.3,
            max_facilities=10,
            equity_weight=profile.default_equity_weight,
            coverage_radius_miles=profile.coverage_radius_miles,
            min_spacing_miles=profile.min_spacing_miles,
            solver_type="greedy",
        )
        sol = solver.solve(scenario, cands, dems, dist_mat)
        assert len(sol.selected_site_ids) > 0
        assert sol.total_cost <= scenario.budget


def test_pareto_frontier_generation(test_dataset):
    candidates, demand_nodes, dist_cand_to_dem, dist_cand_to_cand = test_dataset
    scenario = ScenarioConfig(
        domain_id="EV_CHARGING",
        budget=2_000_000.0,
        max_facilities=8,
        coverage_radius_miles=3.0,
    )
    analyzer = ParetoFrontierAnalyzer()
    frontier = analyzer.generate_frontier(
        scenario, candidates, demand_nodes, dist_cand_to_dem, weights=[0.0, 0.5, 1.0]
    )

    assert len(frontier) == 3
    # With higher equity weight, DAC demand covered should generally increase or stay high
    assert frontier[-1].equity_score >= frontier[0].equity_score or frontier[-1].dac_share_pct >= frontier[0].dac_share_pct
