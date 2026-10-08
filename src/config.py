"""
Configuration parameters, sector-specific profiles, and optimization bounds.
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class SectorProfile:
    sector_id: str
    name: str
    short_title: str
    icon: str
    description: str
    demand_metric_name: str
    demand_components: List[str]
    equity_metric_name: str
    equity_components: List[str]
    policy_reference: str
    default_budget: float
    min_budget: float
    max_budget: float
    default_max_facilities: int
    coverage_radius_miles: float
    min_spacing_miles: float
    unit_cost_min: float
    unit_cost_max: float
    default_equity_weight: float
    default_min_equity_share: float


SECTOR_PROFILES: Dict[str, SectorProfile] = {
    "EV_CHARGING": SectorProfile(
        sector_id="EV_CHARGING",
        name="Electric Vehicle DC Fast Charging Deployment",
        short_title="EV Charging",
        icon="⚡",
        description="Allocates public DC fast chargers balancing highway traffic corridor volume, commercial destination density, and accessibility for underserved areas.",
        demand_metric_name="Traffic Volume & Commercial Activity",
        demand_components=["FDOT 2023 Observed AADT", "Employment & Retail POI Density"],
        equity_metric_name="Designated Disadvantaged Communities",
        equity_components=["Low-income Census tracts", "High transportation cost burden"],
        policy_reference="National Electric Vehicle Infrastructure (NEVI) formula standards & state EV plans",
        default_budget=10_000_000.0,
        min_budget=2_000_000.0,
        max_budget=25_000_000.0,
        default_max_facilities=50,
        coverage_radius_miles=2.5,
        min_spacing_miles=1.0,
        unit_cost_min=160_000.0,
        unit_cost_max=320_000.0,
        default_equity_weight=0.35,
        default_min_equity_share=0.30,
    ),
    "TRANSIT_STOPS": SectorProfile(
        sector_id="TRANSIT_STOPS",
        name="Transit Capital Investment & BRT Station Modernization",
        short_title="Transit Stops",
        icon="🚌",
        description="Prioritizes bus rapid transit (BRT) and local bus stop capital improvements (shelters, ADA boarding pads, real-time arrival signs) in transit-reliant neighborhoods.",
        demand_metric_name="Transit Reliance & Activity Density",
        demand_components=["Zero-vehicle household density", "Job access", "Walkable intersection density"],
        equity_metric_name="Environmental Justice & Low-Income Focus Areas",
        equity_components=["Title VI protected populations", "Households below poverty threshold"],
        policy_reference="FTA Title VI Circular 4702.1B & Section 5307/5339 Capital Investment Programs",
        default_budget=6_000_000.0,
        min_budget=1_000_000.0,
        max_budget=15_000_000.0,
        default_max_facilities=60,
        coverage_radius_miles=0.5,
        min_spacing_miles=0.35,
        unit_cost_min=45_000.0,
        unit_cost_max=130_000.0,
        default_equity_weight=0.45,
        default_min_equity_share=0.40,
    ),
    "SAFETY_VISION_ZERO": SectorProfile(
        sector_id="SAFETY_VISION_ZERO",
        name="Vision Zero High Injury Network Countermeasures",
        description="Allocates proven engineering safety countermeasures (RRFBs, pedestrian refuge islands, signal retrofits) targeting severe crash hot spots.",
        short_title="Safety Countermeasures",
        icon="🛡️",
        demand_metric_name="Severe Crash Exposure & Injury Risk",
        demand_components=["5-Year Fatal & Serious Injury (FSI) crashes", "Pedestrian exposure volume"],
        equity_metric_name="Historically Underserved & Vulnerable Corridors",
        equity_components=["Vulnerable Road Users (VRU) in historically disinvested corridors"],
        policy_reference="FHWA Safe Streets and Roads for All (SS4A) & Highway Safety Improvement Program (HSIP)",
        default_budget=8_000_000.0,
        min_budget=1_500_000.0,
        max_budget=20_000_000.0,
        default_max_facilities=45,
        coverage_radius_miles=0.75,
        min_spacing_miles=0.25,
        unit_cost_min=75_000.0,
        unit_cost_max=240_000.0,
        default_equity_weight=0.40,
        default_min_equity_share=0.35,
    ),
    "BIKE_MICROMOBILITY": SectorProfile(
        sector_id="BIKE_MICROMOBILITY",
        name="Protected Bike Corridors & Micromobility Mobility Hubs",
        short_title="Bike Corridors",
        icon="🚲",
        description="Identifies priority segments for protected bicycle lanes and shared micromobility hubs to close first/last-mile connectivity gaps.",
        demand_metric_name="Active Mobility Latent Demand & Stress Reduction",
        demand_components=["Level of Traffic Stress (LTS) gap", "Commercial/university connectivity"],
        equity_metric_name="Car-Free Mobility Corridors",
        equity_components=["Low-income non-motorized commute sheds"],
        policy_reference="USDOT Reconnecting Communities & Active Transportation Infrastructure Program",
        default_budget=5_000_000.0,
        min_budget=1_000_000.0,
        max_budget=12_000_000.0,
        default_max_facilities=40,
        coverage_radius_miles=1.0,
        min_spacing_miles=0.4,
        unit_cost_min=65_000.0,
        unit_cost_max=175_000.0,
        default_equity_weight=0.35,
        default_min_equity_share=0.35,
    ),
}

# Alias for backward compatibility
DOMAIN_PROFILES = SECTOR_PROFILES

# Data Provenance Metadata
DATA_PROVENANCE = {
    "geography": "Florida (Central Florida / Orlando & Tampa Bay Metros)",
    "fdot_roadway": "Florida Department of Transportation (FDOT) Roadway Characteristics Inventory (RCI) & 2023 Observed AADT",
    "census_demographics": "US Census Bureau / American Community Survey (ACS) 5-Year Estimates (2018-2022)",
    "crash_records": "Florida Signal Four Analytics / FDOT CAR 5-Year Fatal & Serious Injury (FSI) Crashes (2019-2023)",
    "equity_designation": "Disadvantaged Community (DAC) criteria: Poverty rate > 18% or Zero-car household share > 11.5%",
    "candidate_sites_count": 2413,
    "demand_nodes_count": 1200,
}

# Solver Parameters
MIP_TIME_LIMIT_SEC = 5.0
MIP_REL_GAP = 0.015
EARTH_RADIUS_MILES = 3958.8
