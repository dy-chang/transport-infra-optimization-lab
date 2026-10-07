"""
Configuration parameters, domain profiles, and optimization bounds.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class DomainProfile:
    domain_id: str
    name: str
    description: str
    demand_metric_name: str
    equity_metric_name: str
    federal_program: str
    default_budget: float
    min_budget: float
    max_budget: float
    default_max_facilities: int
    coverage_radius_miles: float
    min_spacing_miles: float
    unit_cost_min: float
    unit_cost_max: float
    default_equity_weight: float
    default_equity_quota: float


DOMAIN_PROFILES: Dict[str, DomainProfile] = {
    "EV_CHARGING": DomainProfile(
        domain_id="EV_CHARGING",
        name="Electric Vehicle DC Fast Charging Deployment",
        description="NEVI Formula Program compliant DCFC facility allocation balancing traffic corridor demand and Justice40 equity.",
        demand_metric_name="Traffic AADT & Commercial Density",
        equity_metric_name="Justice40 Disadvantaged Community (DAC) Status",
        federal_program="NEVI Formula Program (23 CFR Part 680)",
        default_budget=10_000_000.0,
        min_budget=2_000_000.0,
        max_budget=25_000_000.0,
        default_max_facilities=50,
        coverage_radius_miles=2.5,
        min_spacing_miles=1.0,
        unit_cost_min=150_000.0,
        unit_cost_max=320_000.0,
        default_equity_weight=0.35,
        default_equity_quota=0.30,
    ),
    "TRANSIT_STOPS": DomainProfile(
        domain_id="TRANSIT_STOPS",
        name="Transit Capital Investment & BRT Station Modernization",
        description="FTA Title VI compliant station capital improvements targeting zero-car households and transit-dependent corridors.",
        demand_metric_name="Transit Ridership Propensity & Activity Density",
        equity_metric_name="Title VI Environmental Justice & Zero-Car Households",
        federal_program="FTA Section 5307 / 5339 Capital Investment Grants",
        default_budget=6_000_000.0,
        min_budget=1_000_000.0,
        max_budget=15_000_000.0,
        default_max_facilities=60,
        coverage_radius_miles=0.5,
        min_spacing_miles=0.35,
        unit_cost_min=50_000.0,
        unit_cost_max=140_000.0,
        default_equity_weight=0.45,
        default_equity_quota=0.40,
    ),
    "SAFETY_VISION_ZERO": DomainProfile(
        domain_id="SAFETY_VISION_ZERO",
        name="Vision Zero High Injury Network Countermeasures",
        description="Systemic safety improvements (RRFBs, curb extensions, signalization) targeting severe crash clusters in vulnerable areas.",
        demand_metric_name="5-Year Fatal & Serious Injury (FSI) Crash Weight",
        equity_metric_name="Vulnerable Road Users (VRU) in Historically Underserved Zones",
        federal_program="FHWA Safe Streets and Roads for All (SS4A)",
        default_budget=8_000_000.0,
        min_budget=1_500_000.0,
        max_budget=20_000_000.0,
        default_max_facilities=45,
        coverage_radius_miles=0.75,
        min_spacing_miles=0.25,
        unit_cost_min=80_000.0,
        unit_cost_max=250_000.0,
        default_equity_weight=0.40,
        default_equity_quota=0.35,
    ),
    "BIKE_MICROMOBILITY": DomainProfile(
        domain_id="BIKE_MICROMOBILITY",
        name="Protected Bike Corridors & Micromobility Mobility Hubs",
        description="Active transportation capital investment reducing Level of Traffic Stress (LTS) in car-free commute sheds.",
        demand_metric_name="Active Mobility Latent Demand & LTS Reduction",
        equity_metric_name="Low-Income Car-Free Commute Corridors",
        federal_program="USDOT Reconnecting Communities & Active Transportation",
        default_budget=5_000_000.0,
        min_budget=1_000_000.0,
        max_budget=12_000_000.0,
        default_max_facilities=40,
        coverage_radius_miles=1.0,
        min_spacing_miles=0.4,
        unit_cost_min=70_000.0,
        unit_cost_max=180_000.0,
        default_equity_weight=0.35,
        default_equity_quota=0.35,
    ),
}

# Solvers Settings
MIP_TIME_LIMIT_SEC = 5.0
MIP_REL_GAP = 0.01  # 1% optimality gap tolerance for fast interactive solving
MILES_TO_METERS = 1609.34
EARTH_RADIUS_MILES = 3958.8
