"""
CLI Example: Run multi-objective Pareto Frontier analysis across transportation sectors.
"""

from pathlib import Path
import sys

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DOMAIN_PROFILES
from src.data_loader import InfrastructureDataLoader
from src.models import ScenarioConfig
from src.solvers.pareto_frontier import ParetoFrontierAnalyzer


def main():
    print("=== Transportation Infrastructure Optimization Lab ===")
    print("Running Multi-Objective Pareto Frontier Analysis across Sectors\n")

    loader = InfrastructureDataLoader(seed=42)
    analyzer = ParetoFrontierAnalyzer()

    for domain_id, profile in DOMAIN_PROFILES.items():
        print(f"--- Sector: {profile.name} ---")
        candidates, demand_nodes, dist_cand_to_dem, dist_cand_to_cand = loader.generate_testbed(
            domain_id=domain_id, n_candidates=1500, n_demand_nodes=800
        )

        scenario = ScenarioConfig(
            domain_id=domain_id,
            budget=profile.default_budget,
            max_facilities=profile.default_max_facilities,
            coverage_radius_miles=profile.coverage_radius_miles,
            min_spacing_miles=profile.min_spacing_miles,
        )

        frontier = analyzer.generate_frontier(
            scenario, candidates, demand_nodes, dist_cand_to_dem, weights=[0.0, 0.25, 0.5, 0.75, 1.0]
        )

        print(f"{'Equity Weight':<15} | {'Total Demand':<15} | {'DAC Demand':<15} | {'DAC Share %':<15} | {'Cost ($M)':<12}")
        print("-" * 80)
        for p in frontier:
            print(f"{p.equity_weight:<15.2f} | {p.efficiency_score:<15.1f} | {p.equity_score:<15.1f} | {p.dac_share_pct:<15.1f}% | ${p.total_cost/1_000_000:<10.2f}M")
        print()


if __name__ == "__main__":
    main()
