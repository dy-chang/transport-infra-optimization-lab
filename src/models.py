"""
Pydantic and dataclass models for optimization entities, scenarios, and solutions.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


@dataclass
class CandidateSite:
    site_id: str
    name: str
    latitude: float
    longitude: float
    cost: float
    sector_id: str
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DemandNode:
    node_id: str
    name: str
    latitude: float
    longitude: float
    demand: float
    is_dac: bool  # Designated Disadvantaged Community
    attributes: Dict[str, Any] = field(default_factory=dict)


class ScenarioConfig(BaseModel):
    sector_id: str = "EV_CHARGING"
    budget: float = Field(default=10_000_000.0, ge=0.0)
    max_facilities: int = Field(default=50, ge=1, le=500)
    equity_weight: float = Field(default=0.35, ge=0.0, le=1.0)
    min_equity_share: float = Field(default=0.30, ge=0.0, le=1.0)
    coverage_radius_miles: float = Field(default=2.5, gt=0.0)
    min_spacing_miles: float = Field(default=1.0, ge=0.0)
    solver_type: str = Field(default="greedy", pattern="^(highs|greedy)$")
    cost_sensitive_mode: bool = False
    cost_penalty_lambda: float = Field(default=1e-7, ge=0.0)

    @property
    def domain_id(self) -> str:
        """Backward compatibility alias for sector_id."""
        return self.sector_id


@dataclass
class OptimizationSolution:
    scenario: ScenarioConfig
    selected_site_ids: List[str]
    total_cost: float
    budget_utilization_pct: float
    total_demand_covered: float
    pct_total_demand_covered: float
    dac_demand_covered: float
    dac_share_pct: float
    is_equity_share_met: bool
    solver_name: str
    solve_time_sec: float
    objective_value: float
    norm_efficiency: float = 0.0
    norm_equity: float = 0.0
    best_bound: Optional[float] = None
    mip_gap: Optional[float] = None
    is_proven_optimal: bool = False
    covered_node_ids: List[str] = field(default_factory=list)
    optimality_gap_pct: Optional[float] = None

    @property
    def is_equity_quota_met(self) -> bool:
        """Backward compatibility alias for is_equity_share_met."""
        return self.is_equity_share_met


@dataclass
class ParetoPoint:
    equity_weight: float
    efficiency_score: float
    equity_score: float
    total_cost: float
    selected_count: int
    dac_share_pct: float


@dataclass
class SensitivityResult:
    parameter_name: str
    parameter_value: Any
    total_demand_covered: float
    dac_demand_covered: float
    dac_share_pct: float
    total_cost: float
    facility_count: int
