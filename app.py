"""
Transportation Infrastructure Optimization Lab
Interactive Streamlit & PyDeck workbench for multi-objective facility location optimization.
"""

from pathlib import Path
import sys
import time

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

from src.config import DOMAIN_PROFILES, DomainProfile
from src.data_loader import InfrastructureDataLoader
from src.models import ScenarioConfig
from src.solvers.greedy_solver import GreedyCoverageSolver
from src.solvers.highs_solver import HiGHSMIPSolver
from src.solvers.pareto_frontier import ParetoFrontierAnalyzer
from src.analytics import compute_optimality_gap, evaluate_equity_fairness

st.set_page_config(
    page_title="Transportation Infrastructure Optimization Lab",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .metric-container {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 14px;
        border-left: 4px solid #1f77b4;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 12px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
</style>
""", unsafe_allow_html=True)

st.title("⚡ Transportation Infrastructure Optimization Lab")
st.markdown("""
**Operations Research • Mathematical Programming (MIP) • Spatial GIS Capital Allocation**  
Solves the Multi-Objective Maximum Coverage Location Problem (MCLP) across **EV Charging**, **Transit Stops**, **Vision Zero Safety**, and **Bike Networks**.
""")

# Sidebar Controls
st.sidebar.header("🎯 Sector & Scenario Controls")

sector_icons = {
    "EV_CHARGING": "⚡ EV Charging (NEVI)",
    "TRANSIT_STOPS": "🚌 Transit Stops (FTA)",
    "SAFETY_VISION_ZERO": "🛡️ Safety Vision Zero (SS4A)",
    "BIKE_MICROMOBILITY": "🚲 Bike Infrastructure",
}

domain_id = st.sidebar.selectbox(
    "Infrastructure Sector:",
    options=list(DOMAIN_PROFILES.keys()),
    format_func=lambda x: sector_icons.get(x, x),
)
profile = DOMAIN_PROFILES[domain_id]
st.sidebar.caption(f"**Program**: {profile.federal_program}")

# Sliders
st.sidebar.subheader("💰 Capital & Cardinality")
budget = st.sidebar.slider(
    "Investment Budget ($M):",
    min_value=float(profile.min_budget / 1_000_000),
    max_value=float(profile.max_budget / 1_000_000),
    value=float(profile.default_budget / 1_000_000),
    step=0.5,
    format="$%.1fM"
) * 1_000_000.0

max_facilities = st.sidebar.slider(
    "Max Facilities to Fund:",
    min_value=5,
    max_value=120,
    value=profile.default_max_facilities,
    step=5,
)

st.sidebar.subheader("⚖️ Multi-Objective Equity")
equity_weight = st.sidebar.slider(
    "Equity Priority Weight (w_eq):",
    min_value=0.0,
    max_value=1.0,
    value=profile.default_equity_weight,
    step=0.05,
    help="0.0 = Pure System Demand Efficiency | 1.0 = Pure Disadvantaged Community (DAC) Prioritization"
)

min_equity_quota = st.sidebar.slider(
    "Justice40 Minimum Quota (%):",
    min_value=0.0,
    max_value=50.0,
    value=profile.default_equity_quota * 100.0,
    step=5.0,
    format="%d%%",
    help="Federal requirement that a minimum share of total benefits accrue to disadvantaged communities."
) / 100.0

st.sidebar.subheader("📐 Spatial Constraints")
cov_radius = st.sidebar.slider(
    "Service Coverage Radius (miles):",
    min_value=0.3,
    max_value=5.0,
    value=profile.coverage_radius_miles,
    step=0.1,
)

min_spacing = st.sidebar.slider(
    "Min Spacing (Anti-Cannibalization, miles):",
    min_value=0.0,
    max_value=3.0,
    value=profile.min_spacing_miles,
    step=0.1,
)

solver_mode = st.sidebar.radio(
    "Solver Engine:",
    options=["greedy", "highs"],
    format_func=lambda x: "⚡ Fast Submodular Greedy (<30ms)" if x == "greedy" else "🎯 Exact HiGHS MIP (Global Optimum)"
)


@st.cache_data(show_spinner="Loading real Florida transportation corridors & candidate sites...")
def get_cached_testbed(d_id: str):
    loader = InfrastructureDataLoader(seed=42)
    return loader.generate_testbed(domain_id=d_id, n_candidates=2413, n_demand_nodes=1200)


candidates, demand_nodes, dist_cand_to_dem, dist_cand_to_cand = get_cached_testbed(domain_id)

scenario = ScenarioConfig(
    domain_id=domain_id,
    budget=budget,
    max_facilities=max_facilities,
    equity_weight=equity_weight,
    min_equity_quota=min_equity_quota,
    coverage_radius_miles=cov_radius,
    min_spacing_miles=min_spacing,
    solver_type=solver_mode,
)

# Execute Solver
t_start = time.time()
if solver_mode == "highs":
    solver = HiGHSMIPSolver()
else:
    solver = GreedyCoverageSolver()

solution = solver.solve(scenario, candidates, demand_nodes, dist_cand_to_dem, dist_cand_to_cand)

# Top KPI Metric Banners
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.metric(
        label="Capital Invested",
        value=f"${solution.total_cost / 1_000_000:.2f}M",
        delta=f"{solution.budget_utilization_pct:.1f}% of ${budget/1_000_000:.1f}M Budget"
    )
with k2:
    st.metric(
        label="Facilities Selected",
        value=f"{len(solution.selected_site_ids)} / {max_facilities}",
        delta=f"Pool: {len(candidates):,} Candidates"
    )
with k3:
    st.metric(
        label="System Demand Covered",
        value=f"{solution.pct_total_demand_covered:.1f}%",
        delta=f"{solution.total_demand_covered:,.0f} Total Units"
    )
with k4:
    equity_status = "normal" if solution.is_equity_quota_met else "inverse"
    st.metric(
        label="Justice40 DAC Share",
        value=f"{solution.dac_share_pct:.1f}%",
        delta=f"Target: ≥{min_equity_quota*100:.0f}% ({'PASS' if solution.is_equity_quota_met else 'DEFICIT'})",
        delta_color=equity_status
    )
with k5:
    st.metric(
        label="Solver Latency",
        value=f"{solution.solve_time_sec * 1000:.1f} ms",
        delta=solution.solver_name.split()[0]
    )

st.markdown("---")

# Main Dashboard Tabs
tab_map, tab_pareto, tab_comparison, tab_table = st.tabs([
    "🗺️ Interactive Spatial Allocation (PyDeck 3D)",
    "📈 Pareto Frontier & Equity Trade-Off",
    "⚖️ OR Solver Comparison (MIP vs Greedy)",
    "📋 Funded Project Capital Program"
])

selected_set = set(solution.selected_site_ids)
covered_set = set(solution.covered_node_ids)

with tab_map:
    st.subheader(f"Optimal Infrastructure Deployment: {profile.name}")

    # Build PyDeck DataFrame for Candidates
    cand_df = pd.DataFrame([
        {
            "site_id": c.site_id,
            "name": c.name,
            "lat": c.latitude,
            "lon": c.longitude,
            "cost": c.cost,
            "cost_str": f"${c.cost:,.0f}",
            "is_selected": c.site_id in selected_set,
            "radius": 450 if c.site_id in selected_set else 180,
            "elevation": 1200 if c.site_id in selected_set else 50,
            "color": [245, 175, 25, 230] if c.site_id in selected_set else [140, 140, 140, 90],
        }
        for c in candidates
    ])

    # Build PyDeck DataFrame for Demand Centroids
    dem_df = pd.DataFrame([
        {
            "node_id": d.node_id,
            "lat": d.latitude,
            "lon": d.longitude,
            "demand": d.demand,
            "is_dac": d.is_dac,
            "is_covered": d.node_id in covered_set,
            # Cyan for Justice40 DAC, Orange for non-DAC
            "color": [0, 200, 255, 180] if d.is_dac else [255, 100, 50, 140],
            "radius": 220 if d.is_covered else 120,
        }
        for d in demand_nodes
    ])

    selected_cands = cand_df[cand_df["is_selected"]]
    unselected_cands = cand_df[~cand_df["is_selected"]]

    mean_lat = float(selected_cands["lat"].mean()) if not selected_cands.empty else float(cand_df["lat"].mean())
    mean_lon = float(selected_cands["lon"].mean()) if not selected_cands.empty else float(cand_df["lon"].mean())

    # Candidate layers
    sel_layer = pdk.Layer(
        "ColumnLayer",
        data=selected_cands,
        get_position=["lon", "lat"],
        get_elevation="elevation",
        elevation_scale=1.0,
        radius=500,
        get_fill_color=[245, 180, 20, 240],
        pickable=True,
        auto_highlight=True,
    )

    unsel_layer = pdk.Layer(
        "ScatterplotLayer",
        data=unselected_cands,
        get_position=["lon", "lat"],
        get_radius=120,
        get_fill_color=[160, 160, 160, 100],
        pickable=True,
    )

    demand_layer = pdk.Layer(
        "ScatterplotLayer",
        data=dem_df,
        get_position=["lon", "lat"],
        get_radius="radius",
        get_fill_color="color",
        pickable=True,
    )

    view_state = pdk.ViewState(
        latitude=mean_lat,
        longitude=mean_lon,
        zoom=10.0,
        pitch=40,
        bearing=10,
    )

    deck = pdk.Deck(
        layers=[demand_layer, unsel_layer, sel_layer],
        initial_view_state=view_state,
        tooltip={
            "html": "<b>{name}</b><br/>Cost: {cost_str}<br/>Status: <b>{'SELECTED' if is_selected else 'Candidate'}</b>"
        },
        map_style="mapbox://styles/mapbox/light-v10",
    )
    st.pydeck_chart(deck)

    # Legend & Caption
    c_leg1, c_leg2, c_leg3 = st.columns(3)
    with c_leg1:
        st.markdown("🟡 **Selected Facilities** (Gold Columns, Funded)")
    with c_leg2:
        st.markdown("🔵 **Justice40 DAC Equity Zones** (Cyan Dots)")
    with c_leg3:
        st.markdown("⚪ **Unselected Candidates** (Gray Dots)")

with tab_pareto:
    st.subheader("Multi-Objective Pareto Trade-Off Frontier")
    st.markdown("""
    Traces the empirical mathematical trade-off between **Total System Efficiency** (Total Demand Covered) 
    and **Federal Equity** (Justice40 DAC Demand Covered) across the full parameter spectrum of $w_{\\text{eq}} \\in [0, 1]$.
    """)

    with st.spinner("Sweeping equity weights and generating Pareto Frontier..."):
        analyzer = ParetoFrontierAnalyzer()
        frontier = analyzer.generate_frontier(
            scenario, candidates, demand_nodes, dist_cand_to_dem, dist_cand_to_cand
        )

    p_df = pd.DataFrame([
        {
            "Equity Weight": p.equity_weight,
            "Total Demand Covered": p.efficiency_score,
            "Justice40 DAC Demand": p.equity_score,
            "DAC Share %": p.dac_share_pct,
            "Total Cost ($M)": p.total_cost / 1_000_000.0,
            "Facilities Funded": p.selected_count,
        }
        for p in frontier
    ])

    fig_pareto = px.line(
        p_df,
        x="Total Demand Covered",
        y="Justice40 DAC Demand",
        text="Equity Weight",
        markers=True,
        title="Empirical Pareto Frontier: Efficiency vs Equity Trade-Off",
        labels={"Total Demand Covered": "System Efficiency (Total Demand)", "Justice40 DAC Demand": "Federal Equity (DAC Demand)"}
    )
    fig_pareto.update_traces(textposition="top right")

    # Add current operating point
    fig_pareto.add_trace(go.Scatter(
        x=[solution.total_demand_covered],
        y=[solution.dac_demand_covered],
        mode="markers",
        marker=dict(color="red", size=14, symbol="star"),
        name=f"Current Solution (w={equity_weight:.2f})"
    ))
    fig_pareto.update_layout(height=420)
    st.plotly_chart(fig_pareto, use_container_width=True)

    st.dataframe(p_df, use_container_width=True)

with tab_comparison:
    st.subheader("Operations Research Benchmark: HiGHS Exact MIP vs Submodular Greedy")
    st.markdown("""
    Direct mathematical comparison between global branch-and-cut optimization and the submodular greedy heuristic.
    """)

    if st.button("🚀 Run Comparative Benchmark on Current Scenario"):
        with st.spinner("Solving with HiGHS Branch-and-Cut and Submodular Greedy..."):
            mip_sol = HiGHSMIPSolver().solve(scenario, candidates, demand_nodes, dist_cand_to_dem, dist_cand_to_cand)
            grd_sol = GreedyCoverageSolver().solve(scenario, candidates, demand_nodes, dist_cand_to_dem, dist_cand_to_cand)
            gap = compute_optimality_gap(mip_sol, grd_sol)

            c_comp1, c_comp2, c_comp3 = st.columns(3)
            with c_comp1:
                st.metric(
                    label="HiGHS Objective Value",
                    value=f"{gap['mip_objective']:,.1f}",
                    delta=f"{gap['mip_runtime_sec']:.3f} s"
                )
            with c_comp2:
                st.metric(
                    label="Greedy Objective Value",
                    value=f"{gap['greedy_objective']:,.1f}",
                    delta=f"{gap['greedy_runtime_sec']:.4f} s"
                )
            with c_comp3:
                st.metric(
                    label="Optimality Gap",
                    value=f"{gap['optimality_gap_pct']:.2f}%",
                    delta=f"{gap['speedup_factor']:.0f}x Speedup"
                )

            st.write(f"**Spatial Site Agreement**: {gap['solution_agreement_pct']}% of selected facility locations match identically between the two solvers.")

with tab_table:
    st.subheader("Capital Investment Program: Selected Facility Projects")
    
    sel_sites = [c for c in candidates if c.site_id in selected_set]
    table_df = pd.DataFrame([
        {
            "Project ID": s.site_id,
            "Location / Name": s.name,
            "Latitude": round(s.latitude, 5),
            "Longitude": round(s.longitude, 5),
            "Capital Cost ($)": f"${s.cost:,.0f}",
            "Grid Capacity (kW)": s.attributes.get("grid_capacity_kw", "N/A"),
            "Parking Stalls": s.attributes.get("parking_stalls", "N/A"),
        }
        for s in sel_sites
    ])
    st.dataframe(table_df, use_container_width=True)

    csv_data = table_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Capital Investment Program (CSV)",
        data=csv_data,
        file_name=f"capital_investment_{domain_id}_{int(budget/1_000_000)}M.csv",
        mime="text/csv"
    )
