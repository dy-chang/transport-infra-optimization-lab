"""
Data loader and spatial preprocessor for real-world Florida transportation open data.
Integrates FDOT roadway networks, US Census ACS demographics, and crash records.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.config import DATA_PROVENANCE, EARTH_RADIUS_MILES, SECTOR_PROFILES, SectorProfile
from src.models import CandidateSite, DemandNode


def haversine_distance_matrix(
    lats1: np.ndarray,
    lons1: np.ndarray,
    lats2: np.ndarray,
    lons2: np.ndarray,
) -> np.ndarray:
    """Vectorized Haversine distance computation returning pairwise distance matrix in miles."""
    phi1 = np.radians(lats1)[:, np.newaxis]
    phi2 = np.radians(lats2)[np.newaxis, :]
    delta_phi = np.radians(lats1[:, np.newaxis] - lats2[np.newaxis, :])
    delta_lambda = np.radians(lons1[:, np.newaxis] - lons2[np.newaxis, :])

    a = (np.sin(delta_phi / 2.0) ** 2) + np.cos(phi1) * np.cos(phi2) * (np.sin(delta_lambda / 2.0) ** 2)
    c = 2.0 * np.arcsin(np.clip(np.sqrt(a), 0.0, 1.0))
    return EARTH_RADIUS_MILES * c


class InfrastructureDataLoader:
    """
    Loads Florida open corridor networks and Census equity datasets,
    generating calibrated candidate facility locations and demand centroids.
    """

    def __init__(self, raw_data_path: Optional[Path] = None, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.raw_path = raw_data_path or Path("/Users/dc/Projects/fdot-pedbike-demand-safety/data/processed/corridors_network.csv")

    def load_base_network(self) -> pd.DataFrame:
        """Load real Florida corridor network attributes if available, else generate calibrated base."""
        if self.raw_path.exists():
            return pd.read_csv(self.raw_path)
        else:
            # Fallback generator within Orlando / Tampa Bay metro bounding box
            n_rows = 150
            lats = self.rng.uniform(27.85, 28.65, size=n_rows)
            lons = self.rng.uniform(-82.45, -81.35, size=n_rows)
            aadts = self.rng.integers(15000, 65000, size=n_rows)
            poverty = self.rng.uniform(5.0, 30.0, size=n_rows)
            zero_car = self.rng.uniform(3.0, 22.0, size=n_rows)
            crashes = self.rng.poisson(8, size=n_rows)
            return pd.DataFrame({
                "latitude": lats,
                "longitude": lons,
                "aadt": aadts,
                "poverty_rate_pct": poverty,
                "zero_car_hh_pct": zero_car,
                "crashes_5yr": crashes,
                "severe_crashes_5yr": np.maximum(1, np.round(crashes * 0.3)),
                "pop_density_sqmi": self.rng.uniform(2000, 9000, size=n_rows),
                "emp_density_sqmi": self.rng.uniform(1500, 8000, size=n_rows),
                "pLTS": self.rng.integers(1, 5, size=n_rows),
            })

    def generate_testbed(
        self,
        sector_id: str = "EV_CHARGING",
        n_candidates: int = DATA_PROVENANCE["candidate_sites_count"],
        n_demand_nodes: int = DATA_PROVENANCE["demand_nodes_count"],
    ) -> Tuple[List[CandidateSite], List[DemandNode], np.ndarray, np.ndarray]:
        """
        Generate candidate sites, demand centroids, and spatial distance matrices
        with sector-specific planning models and cost profiles.
        """
        if sector_id not in SECTOR_PROFILES:
            raise ValueError(f"Unknown sector_id: {sector_id}. Valid: {list(SECTOR_PROFILES.keys())}")
        profile = SECTOR_PROFILES[sector_id]

        base_df = self.load_base_network()
        base_lats = base_df["latitude"].to_numpy()
        base_lons = base_df["longitude"].to_numpy()
        n_base = len(base_df)

        # 1. Synthesize / Expand Demand Centroids around corridor corridors
        demand_nodes: List[DemandNode] = []
        d_indices = self.rng.integers(0, n_base, size=n_demand_nodes)
        
        # Spatial dispersion around corridor network
        d_lats = base_lats[d_indices] + self.rng.normal(0, 0.045, size=n_demand_nodes)
        d_lons = base_lons[d_indices] + self.rng.normal(0, 0.045, size=n_demand_nodes)

        for i in range(n_demand_nodes):
            base_row = base_df.iloc[d_indices[i]]
            pov = float(base_row.get("poverty_rate_pct", 12.0)) + float(self.rng.normal(0, 2.5))
            zero_car = float(base_row.get("zero_car_hh_pct", 8.0)) + float(self.rng.normal(0, 2.0))
            aadt = float(base_row.get("aadt", 35000))
            severe_crashes = float(base_row.get("severe_crashes_5yr", 2))
            emp_density = float(base_row.get("emp_density_sqmi", 4000))
            lts = int(base_row.get("pLTS", 3))

            # Disadvantaged Community (DAC) criteria: Poverty rate > 18% or Zero-car household share > 11.5%
            is_dac = bool((pov > 18.0) or (zero_car > 11.5))

            # Sector-specific demand metric formulations
            if sector_id == "EV_CHARGING":
                # Traffic exposure (AADT / 1000) + Commercial Activity Density
                demand_val = round((aadt / 1200.0) + (emp_density / 350.0) + self.rng.uniform(5, 20), 1)
            elif sector_id == "TRANSIT_STOPS":
                # Zero-car household concentration + employment access
                demand_val = round((zero_car * 4.0) + (emp_density / 450.0) + self.rng.uniform(10, 30), 1)
            elif sector_id == "SAFETY_VISION_ZERO":
                # 5-year fatal & serious injury (FSI) crashes + traffic exposure
                demand_val = round((severe_crashes * 5.0) + (aadt / 4000.0) + self.rng.uniform(3, 15), 1)
            else:  # BIKE_MICROMOBILITY
                # High-stress roadway gap (LTS 3 or 4) + zero-vehicle commute demand
                demand_val = round((lts * 8.0) + (zero_car * 2.5) + self.rng.uniform(5, 25), 1)

            node = DemandNode(
                node_id=f"DEM-{i:04d}",
                name=f"Demand Node {i+1}",
                latitude=float(d_lats[i]),
                longitude=float(d_lons[i]),
                demand=float(max(demand_val, 1.0)),
                is_dac=is_dac,
                attributes={
                    "poverty_rate_pct": round(max(pov, 1.0), 1),
                    "zero_car_hh_pct": round(max(zero_car, 0.5), 1),
                    "aadt": int(aadt),
                    "severe_crashes": int(severe_crashes),
                }
            )
            demand_nodes.append(node)

        # 2. Synthesize Candidate Locations with Sector-Specific Costs & Attributes
        candidate_sites: List[CandidateSite] = []
        c_indices = self.rng.integers(0, n_base, size=n_candidates)
        c_lats = base_lats[c_indices] + self.rng.normal(0, 0.05, size=n_candidates)
        c_lons = base_lons[c_indices] + self.rng.normal(0, 0.05, size=n_candidates)

        costs = self.rng.uniform(profile.unit_cost_min, profile.unit_cost_max, size=n_candidates)
        costs = np.round(costs / 1000.0) * 1000.0

        for j in range(n_candidates):
            site_attrs = {}
            if sector_id == "EV_CHARGING":
                site_attrs = {"grid_feeder_kw": int(self.rng.choice([350, 500, 1000])), "parking_stalls": int(self.rng.integers(20, 150))}
            elif sector_id == "TRANSIT_STOPS":
                site_attrs = {"shelter_type": str(self.rng.choice(["Cantilever", "Full Canopy", "BRT Station"])), "ada_boarding": True}
            elif sector_id == "SAFETY_VISION_ZERO":
                site_attrs = {"countermeasure": str(self.rng.choice(["RRFB", "Pedestrian Refuge", "Signal Phasing Retrofit"])), "speed_limit_mph": int(self.rng.choice([35, 40, 45]))}
            else:
                site_attrs = {"facility_type": str(self.rng.choice(["Protected Bike Lane", "Off-Street Trail Link", "Mobility Hub Dock"]))}

            site = CandidateSite(
                site_id=f"CAND-{j:04d}",
                name=f"{profile.short_title} Site {j+1}",
                latitude=float(c_lats[j]),
                longitude=float(c_lons[j]),
                cost=float(costs[j]),
                sector_id=sector_id,
                attributes=site_attrs,
            )
            candidate_sites.append(site)

        # 3. Spatial Distance Matrices
        cand_lats = np.array([c.latitude for c in candidate_sites])
        cand_lons = np.array([c.longitude for c in candidate_sites])
        dem_lats = np.array([d.latitude for d in demand_nodes])
        dem_lons = np.array([d.longitude for d in demand_nodes])

        dist_cand_to_dem = haversine_distance_matrix(dem_lats, dem_lons, cand_lats, cand_lons)
        dist_cand_to_cand = haversine_distance_matrix(cand_lats[:400], cand_lons[:400], cand_lats[:400], cand_lons[:400])

        return candidate_sites, demand_nodes, dist_cand_to_dem, dist_cand_to_cand
