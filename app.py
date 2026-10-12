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

from src.config import DATA_PROVENANCE, SECTOR_PROFILES, SectorProfile
from src.data_loader import InfrastructureDataLoader
from src.models import ScenarioConfig
from src.solvers.greedy_solver import GreedyCoverageSolver
from src.solvers.highs_solver import HiGHSMIPSolver
from src.solvers.pareto_frontier import ParetoFrontierAnalyzer
from src.analytics import compute_optimality_gap, evaluate_equity_fairness, run_sensitivity_analysis

st.set_page_config(
    page_title="Transportation Infrastructure Optimization Lab",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

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

# Header Section
st.title("⚡ Transportation Infrastructure Optimization Lab")
st.markdown("""
**Where should limited transportation investment go?**  
An open-source decision-support framework optimizing capital allocations across **EV Charging**, **Transit Stops**, **Vision Zero Safety**, and **Bike Infrastructure**.
""")

# Sidebar Controls
st.sidebar.header("🎯 Sector & Scenario Controls")

domain_id = st.sidebar.selectbox(
    "Infrastructure Sector:",
    options=list(SECTOR_PROFILES.keys()),
    format_func=lambda x: f"{SECTOR_PROFILES[x].icon} {SECTOR_PROFILES[x].short_title}",
)
profile = SECTOR_PROFILES[domain_id]

# Show Sector Planning Details
with st.sidebar.expander("ℹ️ Sector Planning Model Details", expanded=False):
    st.write(f"**Policy Framework**: {profile.policy_reference}")
    st.write(f"**Demand Metrics**: {', '.join(profile.demand_components)}")
    st.write(f"**Equity Target**: {', '.join(profile.equity_components)}")
    st.write(f"**Unit Cost (Planning Assumptions)**: ${profile.unit_cost_min:,.0f} – ${profile.unit_cost_max:,.0f}")

st.sidebar.subheader("💰 Capital Budget & Facilities")
budget = st.sidebar.slider(
    "Capital Budget ($M):",
    min_value=float(profile.min_budget / 1_000_000),
    max_value=float(profile.max_budget / 1_000_000),
    value=float(profile.default_budget / 1_000_000),
    step=0.5,
    format="$%.1fM"
) * 1_000_000.0

max_facilities = st.sidebar.slider(
    "Max Facilities to Fund (P):",
    min_value=5,
    max_value=100,
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

min_equity_share = st.sidebar.slider(
    "Equity Coverage Constraint: Min DAC Share (α):",
    min_value=0.0,
    max_value=60.0,
    value=profile.default_min_equity_share * 100.0,
    step=5.0,
    format="%d%%",
    help="Configurable minimum share of covered demand attributable to designated disadvantaged communities."
) / 100.0

st.sidebar.subheader("📐 Spatial Constraints")
cov_radius = st.sidebar.slider(
    "Coverage Service Radius (miles):",
    min_value=0.3,
    max_value=5.0,
    value=profile.coverage_radius_miles,
    step=0.1,
)

min_spacing = st.sidebar.slider(
    "Min Facility Spacing (miles):",
    min_value=0.0,
    max_value=3.0,
    value=profile.min_spacing_miles,
    step=0.1,
    help="Anti-cannibalization constraint preventing facility redundancy within close proximity."
)

solver_mode = st.sidebar.radio(
    "Optimization Engine:",
    options=["greedy", "highs"],
    format_func=lambda x: "⚡ Greedy Marginal-Gain Heuristic (Sub-second interactive)" if x == "greedy" else "🎯 HiGHS MILP Solver (Rigorous Branch-and-Cut)"
)


@st.cache_data(show_spinner="Loading Florida corridor testbed (2,413 candidates & 1,200 demand nodes)...")
def get_cached_testbed(sec_id: str):
    loader = InfrastructureDataLoader(seed=42)
    return loader.generate_testbed(sector_id=sec_id, n_candidates=2413, n_demand_nodes=1200)


candidates, demand_nodes, dist_cand_to_dem, dist_cand_to_cand = get_cached_testbed(domain_id)

scenario = ScenarioConfig(
    sector_id=domain_id,
    budget=budget,
    max_facilities=max_facilities,
    equity_weight=equity_weight,
    min_equity_share=min_equity_share,
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
    equity_status = "normal" if solution.is_equity_share_met else "inverse"
    st.metric(
        label="Disadvantaged Community Share",
        value=f"{solution.dac_share_pct:.1f}%",
        delta=f"Target: ≥{min_equity_share*100:.0f}% ({'MET' if solution.is_equity_share_met else 'DEFICIT'})",
        delta_color=equity_status
    )
with k5:
    st.metric(
        label="Solver Latency",
        value=f"{solution.solve_time_sec * 1000:.1f} ms" if solution.solve_time_sec < 1.0 else f"{solution.solve_time_sec:.2f} s",
        delta=solution.solver_name.split()[0]
    )

st.markdown("---")

# Main Dashboard Tabs
tab_map, tab_pareto, tab_sensitivity, tab_comparison, tab_table, tab_provenance = st.tabs([
    "🗺️ Spatial Allocation (PyDeck 3D)",
    "📈 Pareto Frontier (Efficiency vs Equity)",
    "📊 Sensitivity Analysis (Parametric Sweep)",
    "⚖️ HiGHS MILP vs Heuristic Benchmark",
    "📋 Funded Project Program",
    "📚 Data Provenance & Methodology"
])

selected_set = set(solution.selected_site_ids)
covered_set = set(solution.covered_node_ids)

with tab_map:
    st.subheader(f"Optimal Infrastructure Deployment: {profile.name}")

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
        }
        for c in candidates
    ])

    dem_df = pd.DataFrame([
        {
            "node_id": d.node_id,
            "lat": d.latitude,
            "lon": d.longitude,
            "demand": d.demand,
            "is_dac": d.is_dac,
            "is_covered": d.node_id in covered_set,
            "color": [0, 200, 255, 180] if d.is_dac else [255, 100, 50, 140],
            "radius": 220 if d.is_covered else 120,
        }
        for d in demand_nodes
    ])

    selected_cands = cand_df[cand_df["is_selected"]]
    unselected_cands = cand_df[~cand_df["is_selected"]]

    mean_lat = float(selected_cands["lat"].mean()) if not selected_cands.empty else float(cand_df["lat"].mean())
    mean_lon = float(selected_cands["lon"].mean()) if not selected_cands.empty else float(cand_df["lon"].mean())

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

    c_leg1, c_leg2, c_leg3 = st.columns(3)
    with c_leg1:
        st.markdown("🟡 **Selected Facilities** (Gold Columns, Funded)")
    with c_leg2:
        st.markdown("🔵 **Disadvantaged Community Demand** (Cyan Dots)")
    with c_leg3:
        st.markdown("⚪ **Unfunded Candidates** (Gray Dots)")

with tab_pareto:
    st.subheader("Multi-Objective Pareto Trade-Off Frontier")
    st.markdown("""
    Traces the empirical mathematical trade-off between **Total System Efficiency** (Total Demand Covered) 
    and **Equity** (Disadvantaged Community Demand Covered) across the full parameter spectrum of $w_{\\text{eq}} \\in [0, 1]$.
    """)

    with st.spinner("Generating multi-objective Pareto Frontier..."):
        analyzer = ParetoFrontierAnalyzer()
        frontier = analyzer.generate_frontier(
            scenario, candidates, demand_nodes, dist_cand_to_dem, dist_cand_to_cand
        )

    p_df = pd.DataFrame([
        {
            "Equity Weight": p.equity_weight,
            "Total Demand Covered": p.efficiency_score,
            "DAC Demand Covered": p.equity_score,
            "DAC Share %": p.dac_share_pct,
            "Total Cost ($M)": p.total_cost / 1_000_000.0,
            "Facilities Funded": p.selected_count,
        }
        for p in frontier
    ])

    fig_pareto = px.line(
        p_df,
        x="Total Demand Covered",
        y="DAC Demand Covered",
        text="Equity Weight",
        markers=True,
        title="Empirical Pareto Frontier: System Efficiency vs Equity Coverage",
        labels={"Total Demand Covered": "System Efficiency (Total Demand)", "DAC Demand Covered": "Equity (Disadvantaged Community Demand)"}
    )
    fig_pareto.update_traces(textposition="top right")

    # Mark current solution
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

with tab_sensitivity:
    st.subheader("Parametric Sensitivity Analysis")
    st.markdown("""
    Evaluate how infrastructure investment returns vary across budget thresholds, equity priority weights, and coverage radii.
    """)

    sens_col1, sens_col2 = st.columns([1, 2])
    with sens_col1:
        param_to_sweep = st.selectbox(
            "Sweep Parameter:",
            options=["budget", "equity_weight", "coverage_radius_miles", "max_facilities"],
            format_func=lambda x: {
                "budget": "Capital Budget ($)",
                "equity_weight": "Equity Weight (w_eq)",
                "coverage_radius_miles": "Coverage Service Radius (miles)",
                "max_facilities": "Max Facilities Funded (P)"
            }[x]
        )
        sweep_solver = st.radio(
            "Sensitivity Solver:",
            options=["greedy", "highs"],
            format_func=lambda x: "⚡ Greedy (Instant Sweep)" if x == "greedy" else "🎯 HiGHS (Branch-and-Cut Sweep)"
        )
        run_sens_btn = st.button("📊 Run Sensitivity Sweep", type="primary")

    with sens_col2:
        if run_sens_btn:
            with st.spinner("Computing parametric sensitivity sweep..."):
                sens_results = run_sensitivity_analysis(
                    scenario=scenario,
                    candidates=candidates,
                    demand_nodes=demand_nodes,
                    dist_cand_to_dem=dist_cand_to_dem,
                    dist_cand_to_cand=dist_cand_to_cand,
                    parameter_name=param_to_sweep,
                    solver_type=sweep_solver,
                )

                s_df = pd.DataFrame([
                    {
                        "Parameter Value": r.parameter_value if param_to_sweep != "budget" else f"${r.parameter_value/1_000_000:.1f}M",
                        "Raw Value": r.parameter_value,
                        "Total Demand Covered": r.total_demand_covered,
                        "DAC Demand Covered": r.dac_demand_covered,
                        "DAC Share %": r.dac_share_pct,
                        "Total Cost ($M)": r.total_cost / 1_000_000.0,
                        "Funded Facilities": r.facility_count,
                    }
                    for r in sens_results
                ])

                fig_sens = px.line(
                    s_df,
                    x="Raw Value",
                    y=["Total Demand Covered", "DAC Demand Covered"],
                    markers=True,
                    title=f"Coverage Response across {param_to_sweep}",
                    labels={"Raw Value": param_to_sweep, "value": "Demand Covered"},
                )
                st.plotly_chart(fig_sens, use_container_width=True)
                st.dataframe(s_df.drop(columns=["Raw Value"]), use_container_width=True)
        else:
            st.info("Select a parameter and click **Run Sensitivity Sweep** to analyze parameter responsiveness.")

with tab_comparison:
    st.subheader("Operations Research Benchmark: HiGHS MILP vs Greedy Heuristic")
    st.markdown("""
    Direct mathematical comparison between branch-and-cut optimization and the marginal-gain heuristic.  
    *HiGHS MILP provides rigorous planning benchmarks with verifiable solver gap bounds, while the greedy heuristic enables sub-second interactive exploration.*
    """)

    c_bench1, c_bench2 = st.columns([1, 1])
    with c_bench1:
        st.markdown("""
        | Dimension | HiGHS MILP | Greedy Marginal-Gain Heuristic |
        | :--- | :--- | :--- |
        | **Algorithm** | Simplex + Branch-and-Cut | Incremental Benefit-to-Cost Ratio |
        | **500-site benchmark** | Proven optimal (solver gap = 0.0%) | Fast heuristic |
        | **Full network (2,413 sites)** | Best incumbent within time limit | Fast heuristic |
        | **Optimality information** | Dual bound, best incumbent, MIP gap | Empirical gap relative to MILP |
        | **Primary role** | Rigorous planning benchmark | Sub-second interactive exploration |
        """)

    with c_bench2:
        if st.button("🚀 Run Live Solver Comparison on Current Scenario"):
            with st.spinner("Executing HiGHS Branch-and-Cut and Greedy Heuristic..."):
                mip_sol = HiGHSMIPSolver().solve(scenario, candidates, demand_nodes, dist_cand_to_dem, dist_cand_to_cand)
                grd_sol = GreedyCoverageSolver().solve(scenario, candidates, demand_nodes, dist_cand_to_dem, dist_cand_to_cand)
                gap = compute_optimality_gap(mip_sol, grd_sol)

                st.write(f"**HiGHS Runtime**: {mip_sol.solve_time_sec:.3f} s")
                st.write(f"**HiGHS Incumbent Objective**: {mip_sol.objective_value:.4f}")
                if mip_sol.best_bound is not None:
                    st.write(f"**HiGHS Dual Bound**: {mip_sol.best_bound:.4f}")
                if mip_sol.mip_gap is not None:
                    st.write(f"**HiGHS Solver MIP Gap**: {mip_sol.mip_gap * 100:.2f}% ({'Proven Global Optimum' if mip_sol.is_proven_optimal else 'Feasible Incumbent'})")
                st.markdown("---")
                st.write(f"**Greedy Runtime**: {grd_sol.solve_time_sec * 1000:.1f} ms")
                st.write(f"**Greedy Objective**: {grd_sol.objective_value:.4f}")
                st.write(f"**Empirical Gap**: {gap['optimality_gap_pct']:.2f}%")
                st.write(f"**Speedup Factor**: {gap['speedup_factor']:.0f}x faster")
                st.write(f"**Site Agreement**: {gap['solution_agreement_pct']:.1f}% matching locations")

with tab_table:
    st.subheader("Capital Investment Program: Selected Facility Projects")
    sel_sites = [c for c in candidates if c.site_id in selected_set]
    table_df = pd.DataFrame([
        {
            "Project ID": s.site_id,
            "Location Name": s.name,
            "Latitude": round(s.latitude, 5),
            "Longitude": round(s.longitude, 5),
            "Capital Cost ($)": f"${s.cost:,.0f}",
        }
        for s in sel_sites
    ])
    st.dataframe(table_df, use_container_width=True)

    csv_data = table_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Capital Investment Program (CSV)",
        data=csv_data,
        file_name=f"capital_program_{domain_id}_{int(budget/1_000_000)}M.csv",
        mime="text/csv"
    )

with tab_provenance:
    st.subheader("Open Data Provenance & Methodology")
    st.markdown(f"""
    ### Primary Raw Open Data Sources
    - **Roadway Network & Traffic Counts**: {DATA_PROVENANCE['fdot_roadway']}
    - **Socio-Demographics & Vehicle Ownership**: {DATA_PROVENANCE['census_demographics']}
    - **Safety & Crash Records**: {DATA_PROVENANCE['crash_records']}
    - **Equity Screening**: {DATA_PROVENANCE['equity_designation']}
    
    ### Derived Optimization Network
    - **Geographic Boundary**: {DATA_PROVENANCE['geography']}
    - **Derived Candidate Sites**: {DATA_PROVENANCE['candidate_sites_count']:,} candidate locations derived from FDOT road intersections, transit hubs, and commercial corridors.
    - **Derived Demand Nodes**: {DATA_PROVENANCE['demand_nodes_count']:,} demand centroids derived from US Census ACS block groups.
    - **Cost Structure**: {DATA_PROVENANCE['cost_basis']}
    """)
