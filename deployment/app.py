"""
Business Travel CO2 Dashboard
=============================

Design rationale (Bach, 2022; Few, 2006; Sarikaya et al., 2018)
---------------------------------------------------------------
Genre        : Analytic dashboard - decision support
Type         : Tactical (mid-term, mid-level), Organizational audience
Purpose      : Monitor CO2 budget compliance per business unit and surface
               concrete reduction levers for upcoming travel.

Design questions answered (Bach 2022, lecture slide "Questions"):
  Audience  -> Sustainability / travel managers
  Tasks     -> (1) Detect budget overruns at a glance
               (2) Compare BUs
               (3) Identify routes with greener alternatives
  Info      -> Total CO2 vs budget, CO2 by BU, geographic concentration,
               saving potential
  Visualize -> Gauges (status), choropleth/connection map (geography),
               bar (comparison), table (action list)
  Layout    -> Symmetric grid, single-page scrollfit
  Color     -> Semantic only: green=under, amber=approaching, red=over,
               consistent BU palette across all charts
  Interact  -> Sidebar filters, hover tooltips, drill-down via expanders

Applied guidelines (selection from the 20 in the lecture)
  #1  Don't overwhelm  : 4-section narrative, KPIs first
  #4  Carefully chose KPIs : 4 KPIs that map directly to user decisions
  #7  Consistency      : Single font, BU colors fixed across all views
  #9  Manage complexity: Filters in sidebar, detail in collapsed expanders
  #11 Group by attribute: All BU-related views adjacent
  #13 Balance data+space: Generous whitespace, no gridline noise
  #16 Show information, not data: Headlines like "BU3 over budget by 15%"
  #19 State metadata   : Header strip with source, period, scope
  #20 Use color carefully: Status colors only on status, neutral elsewhere

Run
---
    pip install -r requirements.txt
    streamlit run app.py
"""

from io import BytesIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Business Travel CO2 Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Design system (lecture guideline #7 + #20: consistency, color discipline)
# ---------------------------------------------------------------------------
COLOR = {
    "ink":     "#1F2937",
    "muted":   "#6B7280",
    "border":  "#E5E7EB",
    "bg_soft": "#F9FAFB",
    "ok":      "#10B981",
    "warn":    "#F59E0B",
    "bad":     "#EF4444",
}

BU_COLOR = {
    "BU1": "#2563EB",
    "BU2": "#7C3AED",
    "BU3": "#0891B2",
    "BU4": "#DB2777",
}

MODE_COLOR = {
    "flight":     "#EF4444",
    "train":      "#10B981",
    "bus":        "#3B82F6",
    "rental_car": "#F59E0B",
}

CUSTOM_CSS = """
<style>
  html, body, [class*="css"]  { font-family: 'Inter', 'Arial', sans-serif; color: #1F2937; }
  .block-container            { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1400px; }
  h1, h2, h3, h4              { color: #1F2937; font-weight: 600; letter-spacing: -0.01em; }
  h1                          { font-size: 1.65rem; margin-bottom: 0.1rem; }
  .subtitle                   { color: #6B7280; font-size: 0.95rem; margin-bottom: 1rem; }
  .meta-strip                 { background:#F9FAFB; border:1px solid #E5E7EB; border-radius:8px;
                                padding:0.55rem 0.9rem; font-size:0.82rem; color:#4B5563;
                                margin-bottom:1.4rem; display:flex; gap:1.5rem; flex-wrap:wrap; }
  .meta-strip b               { color:#1F2937; }
  .section-title              { font-size:1.05rem; font-weight:600; margin-top:1.6rem;
                                margin-bottom:0.6rem; color:#1F2937;
                                border-bottom:1px solid #E5E7EB; padding-bottom:0.4rem; }
  .section-hint               { color:#6B7280; font-size:0.85rem; margin-top:-0.3rem;
                                margin-bottom:0.9rem; }
  .kpi-card                   { background:#FFFFFF; border:1px solid #E5E7EB; border-radius:10px;
                                padding:1rem 1.1rem; height:100%; }
  .kpi-label                  { color:#6B7280; font-size:0.78rem; font-weight:500;
                                text-transform:uppercase; letter-spacing:0.04em; }
  .kpi-value                  { color:#1F2937; font-size:1.8rem; font-weight:700;
                                margin-top:0.25rem; line-height:1.1; }
  .kpi-delta-ok               { color:#10B981; font-size:0.85rem; font-weight:500; }
  .kpi-delta-bad              { color:#EF4444; font-size:0.85rem; font-weight:500; }
  .kpi-delta-neutral          { color:#6B7280; font-size:0.85rem; }
  .headline-ok                { color:#065F46; background:#ECFDF5; border:1px solid #A7F3D0;
                                border-radius:6px; padding:0.5rem 0.75rem; font-size:0.85rem; }
  .headline-warn              { color:#92400E; background:#FFFBEB; border:1px solid #FDE68A;
                                border-radius:6px; padding:0.5rem 0.75rem; font-size:0.85rem; }
  .headline-bad               { color:#991B1B; background:#FEF2F2; border:1px solid #FECACA;
                                border-radius:6px; padding:0.5rem 0.75rem; font-size:0.85rem; }
  [data-testid="stSidebar"]   { background:#FFFFFF; border-right:1px solid #E5E7EB; }
  [data-testid="stSidebar"] *,
  [data-testid="stSidebar"] label,
  [data-testid="stSidebar"] p,
  [data-testid="stSidebar"] h1,
  [data-testid="stSidebar"] h2,
  [data-testid="stSidebar"] h3,
  [data-testid="stSidebar"] h4 { color:#1F2937 !important; }
  [data-testid="stMetricValue"] { font-size: 1.5rem; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Plotly default look. NOTE: margin is set per-figure to avoid duplicate kwargs.
PLOTLY_BASE = dict(
    font=dict(family="Inter, Arial, sans-serif", size=12, color=COLOR["ink"]),
    paper_bgcolor="white",
    plot_bgcolor="white",
)


def plotly_layout(**overrides):
    """Merge base style with per-figure overrides without duplicating keys."""
    layout = dict(PLOTLY_BASE)
    layout.update(overrides)
    return layout

# ---------------------------------------------------------------------------
# Data layer
# ---------------------------------------------------------------------------
REQUIRED_HIST = [
    "transport_mode", "departure_iata", "arrival_iata",
    "departure_lat", "departure_lon", "arrival_lat", "arrival_lon",
    "business_unit",
]
REQUIRED_INPUT = [
    "transport_mode", "departure_iata", "arrival_iata", "business_unit",
]


@st.cache_data(show_spinner=False)
def load_workbook(file_bytes: bytes) -> dict:
    buf = BytesIO(file_bytes)
    buf.seek(0)
    return pd.read_excel(buf, sheet_name=None)


def parse_budgets(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {}
    col = next((c for c in df.columns if "2026" in str(c)), None)
    if col is None:
        return {}
    s = df[col].astype(str).str.replace(",", ".", regex=False).str.replace(" ", "", regex=False)
    df = df.assign(b=pd.to_numeric(s, errors="coerce"))
    return {bu: v for bu, v in zip(df["Business Unit"], df["b"]) if isinstance(bu, str) and bu.startswith("BU")}


def route_averages(df: pd.DataFrame, co2_col: str) -> pd.DataFrame:
    return (
        df.groupby(["departure_iata", "arrival_iata", "transport_mode"], dropna=False)
        .agg(
            avg_co2=(co2_col, "mean"),
            avg_km=("km", "mean"),
            avg_cost=("cost_CHF", "mean"),
            n_hist=(co2_col, "count"),
            dep_lat=("departure_lat", "first"),
            dep_lon=("departure_lon", "first"),
            arr_lat=("arrival_lat", "first"),
            arr_lon=("arrival_lon", "first"),
        )
        .reset_index()
    )


def enrich_input(input_df: pd.DataFrame, route_avg: pd.DataFrame, co2_col: str) -> pd.DataFrame:
    """Look up CO2 average and coordinates for each input trip."""
    merged = input_df.merge(
        route_avg[
            [
                "departure_iata", "arrival_iata", "transport_mode",
                "avg_co2", "avg_km", "dep_lat", "dep_lon", "arr_lat", "arr_lon",
            ]
        ],
        on=["departure_iata", "arrival_iata", "transport_mode"],
        how="left",
    )
    if co2_col in merged.columns:
        merged["estimated_co2"] = merged[co2_col].fillna(merged["avg_co2"])
    else:
        merged["estimated_co2"] = merged["avg_co2"]
    # Coordinates: prefer existing, fallback to lookup
    for col in ("departure_lat", "departure_lon", "arrival_lat", "arrival_lon"):
        if col not in merged.columns:
            merged[col] = np.nan
    merged["departure_lat"] = merged["departure_lat"].fillna(merged["dep_lat"])
    merged["departure_lon"] = merged["departure_lon"].fillna(merged["dep_lon"])
    merged["arrival_lat"] = merged["arrival_lat"].fillna(merged["arr_lat"])
    merged["arrival_lon"] = merged["arrival_lon"].fillna(merged["arr_lon"])
    if "km" not in merged.columns:
        merged["km"] = merged["avg_km"]
    return merged.drop(columns=["dep_lat", "dep_lon", "arr_lat", "arr_lon"], errors="ignore")


def find_alternatives(estimated: pd.DataFrame, route_avg: pd.DataFrame) -> pd.DataFrame:
    flights = estimated[estimated["transport_mode"] == "flight"].copy()
    # Preserve the original index so apply_alternatives can write back correctly.
    flights = flights.reset_index().rename(columns={"index": "_orig_idx"})
    alt = (
        route_avg[route_avg["transport_mode"] != "flight"][
            ["departure_iata", "arrival_iata", "transport_mode", "avg_co2"]
        ]
        .rename(columns={"transport_mode": "alt_mode", "avg_co2": "alt_co2"})
    )
    merged = flights.merge(alt, on=["departure_iata", "arrival_iata"], how="inner")
    merged = merged.dropna(subset=["estimated_co2", "alt_co2"])
    merged = merged[merged["alt_co2"] < merged["estimated_co2"]].copy()
    merged["saving_t"] = merged["estimated_co2"] - merged["alt_co2"]
    merged["saving_pct"] = merged["saving_t"] / merged["estimated_co2"] * 100
    # Pick best alternative per ORIGINAL trip
    idx = merged.groupby("_orig_idx")["saving_t"].idxmax()
    best = merged.loc[idx].set_index("_orig_idx")
    best.index.name = None
    return best.sort_values("saving_t", ascending=False)


def apply_alternatives(estimated: pd.DataFrame, alts: pd.DataFrame) -> pd.DataFrame:
    """Replace flights with their greener alternative where one exists.

    Returns a new DataFrame where transport_mode and estimated_co2 of the
    affected rows are swapped to the alt_mode/alt_co2 values.
    Adds a column 'mode_shifted' (bool) so the UI can highlight changes.
    """
    out = estimated.copy()
    out["mode_shifted"] = False
    if alts.empty:
        return out
    # alts is indexed by the original row index of estimated -> apply directly
    swap_idx = alts.index
    out.loc[swap_idx, "transport_mode"] = alts["alt_mode"].values
    out.loc[swap_idx, "estimated_co2"] = alts["alt_co2"].values
    out.loc[swap_idx, "mode_shifted"] = True
    return out


# ---------------------------------------------------------------------------
# Visual components
# ---------------------------------------------------------------------------
def gauge(value: float, budget: float, title: str, color: str) -> go.Figure:
    if budget is None or pd.isna(budget) or budget <= 0:
        budget = max(value, 1.0)
    axis_max = max(budget * 1.4, value * 1.05)
    pct = (value / budget * 100) if budget > 0 else 0
    if pct > 100:
        bar = COLOR["bad"]
    elif pct > 85:
        bar = COLOR["warn"]
    else:
        bar = COLOR["ok"]

    # Round axis_max to a clean number for nicer tick labels
    if axis_max < 50:
        nice_step = 10
    elif axis_max < 200:
        nice_step = 25
    else:
        nice_step = 50
    axis_max = int(np.ceil(axis_max / nice_step) * nice_step)
    tick_vals = list(range(0, axis_max + 1, nice_step))

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": " t", "valueformat": ".1f",
                    "font": {"size": 30, "color": COLOR["ink"], "family": "Inter, Arial"}},
            title={
                "text": (
                    f"<span style='font-size:1rem;font-weight:600;color:{color}'>{title}</span>"
                    f"<span style='font-size:0.78rem;color:{COLOR['muted']}'>"
                    f"&nbsp;&nbsp;Budget {budget:.0f} t</span>"
                ),
            },
            gauge={
                "axis": {
                    "range": [0, axis_max],
                    "tickvals": tick_vals,
                    "ticktext": [str(v) for v in tick_vals],
                    "tickwidth": 1, "tickcolor": COLOR["border"],
                    "tickfont": {"size": 9, "color": COLOR["muted"]},
                    "ticklen": 4,
                },
                "bar": {"color": bar, "thickness": 0.28, "line": {"width": 0}},
                "bgcolor": "white",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, budget * 0.85], "color": "#F3F4F6"},
                    {"range": [budget * 0.85, budget], "color": "#FEF3C7"},
                    {"range": [budget, axis_max],   "color": "#FEE2E2"},
                ],
                "threshold": {
                    "line": {"color": COLOR["ink"], "width": 3},
                    "thickness": 0.9,
                    "value": budget,
                },
            },
        )
    )
    fig.update_layout(**plotly_layout(
        height=240, margin=dict(l=20, r=20, t=70, b=20),
    ))
    return fig


def bar_bu_vs_budget(emissions: dict, budgets: dict) -> go.Figure:
    bus = sorted(set(emissions.keys()) | set(budgets.keys()))
    actual = [emissions.get(b, 0) for b in bus]
    budget = [budgets.get(b, np.nan) for b in bus]
    colors = [BU_COLOR.get(b, COLOR["muted"]) for b in bus]

    fig = go.Figure()
    fig.add_bar(
        y=bus, x=actual, orientation="h", name="Actual",
        marker=dict(color=colors), hovertemplate="<b>%{y}</b><br>Actual: %{x:.1f} t<extra></extra>",
    )
    # Budget markers as dashed lines
    for i, (bu, bud) in enumerate(zip(bus, budget)):
        if not pd.isna(bud):
            fig.add_shape(
                type="line", x0=bud, x1=bud, y0=i - 0.4, y1=i + 0.4,
                line=dict(color=COLOR["ink"], width=2, dash="dot"),
            )
    fig.add_trace(
        go.Scatter(
            x=[b for b in budget if not pd.isna(b)],
            y=[bu for bu, b in zip(bus, budget) if not pd.isna(b)],
            mode="markers", marker=dict(symbol="line-ns", size=18, color=COLOR["ink"]),
            name="Budget", hovertemplate="<b>%{y}</b><br>Budget: %{x:.1f} t<extra></extra>",
        )
    )
    fig.update_layout(**plotly_layout(
        height=230, showlegend=True,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", y=1.15, x=0),
        xaxis=dict(title="CO2 (t)", showgrid=True, gridcolor=COLOR["border"], zeroline=False),
        yaxis=dict(showgrid=False, autorange="reversed"),
    ))
    return fig


REGION_VIEWS = {
    "World":     {"projection": "natural earth", "scope": "world",  "center": None,                      "lonaxis": None,            "lataxis": None},
    "Europe":    {"projection": "mercator",      "scope": "europe", "center": {"lon": 10, "lat": 50},     "lonaxis": [-25, 45],       "lataxis": [35, 70]},
    "Americas":  {"projection": "mercator",      "scope": "world",  "center": {"lon": -80, "lat": 20},    "lonaxis": [-130, -30],     "lataxis": [-50, 60]},
    "Asia":      {"projection": "mercator",      "scope": "world",  "center": {"lon": 100, "lat": 30},    "lonaxis": [40, 150],       "lataxis": [-10, 60]},
}


def world_map(routes: pd.DataFrame, region: str = "World") -> go.Figure:
    fig = go.Figure()
    base_layout_kwargs = dict(
        margin=dict(l=0, r=0, t=10, b=0),
        height=620,
        legend=dict(
            orientation="h", y=-0.04, x=0.5, xanchor="center",
            bgcolor="rgba(255,255,255,0.85)", bordercolor=COLOR["border"], borderwidth=1,
            font=dict(size=11),
        ),
    )
    view = REGION_VIEWS.get(region, REGION_VIEWS["World"])
    geo_cfg = dict(
        showland=True, landcolor="#F1F3F5",
        showcountries=True, countrycolor="#CBD5E1", countrywidth=0.5,
        showocean=True, oceancolor="#F8FAFC",
        showcoastlines=True, coastlinecolor="#94A3B8", coastlinewidth=0.5,
        showframe=False, projection_type=view["projection"],
        bgcolor="white",
        showsubunits=False,
    )
    if view["center"]:  geo_cfg["center"] = view["center"]
    if view["lonaxis"]: geo_cfg["lonaxis_range"] = view["lonaxis"]
    if view["lataxis"]: geo_cfg["lataxis_range"] = view["lataxis"]

    if routes.empty:
        fig.update_layout(**plotly_layout(geo=geo_cfg, **base_layout_kwargs))
        return fig

    routes = routes.dropna(subset=["dep_lat", "dep_lon", "arr_lat", "arr_lon"]).copy()
    if routes.empty:
        fig.update_layout(**plotly_layout(geo=geo_cfg, **base_layout_kwargs))
        return fig

    max_co2 = max(routes["total_co2"].max(), 1)

    # Lines per mode with scaled thickness; one trace per route so hover works
    for mode, color in MODE_COLOR.items():
        sub = routes[routes["transport_mode"] == mode]
        if sub.empty:
            continue
        for _, r in sub.iterrows():
            width = 1.0 + (r["total_co2"] / max_co2) * 6.5
            hover = (
                f"<b>{r['departure_iata']} -> {r['arrival_iata']}</b><br>"
                f"Mode: {mode}<br>"
                f"Trips: {int(r['n_trips'])}<br>"
                f"Total CO2: {r['total_co2']:.2f} t"
            )
            fig.add_trace(
                go.Scattergeo(
                    lon=[r["dep_lon"], r["arr_lon"]],
                    lat=[r["dep_lat"], r["arr_lat"]],
                    mode="lines",
                    line=dict(width=width, color=color),
                    opacity=0.7, showlegend=False,
                    hoverinfo="text", hovertext=hover, hoverlabel=dict(bgcolor="white"),
                )
            )
        # Single legend entry per mode (only modes actually present)
        fig.add_trace(
            go.Scattergeo(
                lon=[None], lat=[None], mode="lines",
                line=dict(width=4, color=color),
                name=mode.replace("_", " ").title(), showlegend=True,
            )
        )

    # Airport markers with IATA hover
    points = (
        pd.concat(
            [
                routes[["departure_iata", "dep_lat", "dep_lon"]].rename(
                    columns={"departure_iata": "iata", "dep_lat": "lat", "dep_lon": "lon"}
                ),
                routes[["arrival_iata", "arr_lat", "arr_lon"]].rename(
                    columns={"arrival_iata": "iata", "arr_lat": "lat", "arr_lon": "lon"}
                ),
            ]
        )
        .drop_duplicates(subset="iata").dropna()
    )
    fig.add_trace(
        go.Scattergeo(
            lon=points["lon"], lat=points["lat"], mode="markers",
            marker=dict(size=5, color=COLOR["ink"], line=dict(width=1, color="white")),
            text=points["iata"], hoverinfo="text",
            hoverlabel=dict(bgcolor="white"),
            name="Airports", showlegend=False,
        )
    )
    fig.update_layout(**plotly_layout(geo=geo_cfg, **base_layout_kwargs))
    return fig


# ---------------------------------------------------------------------------
# Sidebar (lecture guideline #9: manage complexity, keep filters separate)
# ---------------------------------------------------------------------------
st.sidebar.markdown("### Data sources")
hist_file = st.sidebar.file_uploader(
    "1. Historical reference (Excel)",
    type=["xlsx"], key="hist",
    help="e.g. traveldataexport_clean.xlsx - provides route averages and budgets",
)
input_file = st.sidebar.file_uploader(
    "2. Planned trips (Excel)",
    type=["xlsx"], key="inp",
    help="e.g. input_data.xlsx - the trips to be analysed",
)

st.sidebar.markdown("### Method")
co2_metric = st.sidebar.radio(
    "CO2 accounting metric",
    ["CO2e RFI2 (t)", "CO2e RFI2.7 (t)"],
    help="RFI = Radiative Forcing Index. RFI 2.7 reflects high-altitude flight effects more strongly.",
)

# ---------------------------------------------------------------------------
# Main canvas
# ---------------------------------------------------------------------------
st.markdown("# Business Travel CO2 Dashboard")
st.markdown(
    "<div class='subtitle'>"
    "Tactical decision support for travel and sustainability managers - "
    "monitor budget compliance and identify reduction levers at a glance."
    "</div>",
    unsafe_allow_html=True,
)


if hist_file is None:
    st.info(
        "Please upload the **historical reference Excel** in the sidebar to begin. "
        "Optionally, also upload a **planned trips file** to analyse upcoming travel; "
        "without it the dashboard analyses the historical data itself."
    )
else:
    # ------------------------------------------------------------------
    # Load historical workbook
    # ------------------------------------------------------------------
    hist_book = load_workbook(hist_file.getvalue())
    if "travel_data" not in hist_book:
        st.error("Sheet 'travel_data' not found in the historical file.")
    else:
        hist = hist_book["travel_data"].copy()
        missing = [c for c in REQUIRED_HIST if c not in hist.columns]
        if missing:
            st.error(f"Historical file is missing columns: {missing}")
        elif co2_metric not in hist.columns:
            st.error(f"CO2 column '{co2_metric}' not found in historical data.")
        else:
            hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
            budgets = parse_budgets(hist_book.get("budget_2026"))
            ravg = route_averages(hist, co2_metric)

            # ------------------------------------------------------------------
            # Decide source: planned trips file or fall back to historical
            # ------------------------------------------------------------------
            _inp_error = None
            if input_file is not None:
                inp_book = load_workbook(input_file.getvalue())
                sheet_name = next(
                    (s for s in ["planned_trips", "travel_data"] if s in inp_book),
                    list(inp_book.keys())[0],
                )
                inp = inp_book[sheet_name].copy()
                miss_in = [c for c in REQUIRED_INPUT if c not in inp.columns]
                if miss_in:
                    _inp_error = f"Input file is missing required columns: {miss_in}"
                else:
                    if "date" in inp.columns:
                        inp["date"] = pd.to_datetime(inp["date"], errors="coerce")
                    src_label = f"Planned trips file ({input_file.name})"
                    n_input = len(inp)
            else:
                inp = hist.copy()
                src_label = "Historical data (no planned trips uploaded)"
                n_input = len(inp)

            if _inp_error:
                st.error(_inp_error)
            else:
                estimated_original = enrich_input(inp, ravg, co2_metric)

                # ------------------------------------------------------------------
                # Scenario state
                # ------------------------------------------------------------------
                if "scenario" not in st.session_state:
                    st.session_state.scenario = "as_planned"

                alts_original = find_alternatives(estimated_original, ravg)
                saving_potential = float(alts_original["saving_t"].sum()) if not alts_original.empty else 0.0

                if st.session_state.scenario == "optimised":
                    estimated = apply_alternatives(estimated_original, alts_original)
                    alts = pd.DataFrame()
                else:
                    estimated = estimated_original
                    alts = alts_original

                # Period info
                if "date" in estimated.columns and estimated["date"].notna().any():
                    d_min = estimated["date"].min()
                    d_max = estimated["date"].max()
                    period = f"{d_min:%Y-%m-%d} to {d_max:%Y-%m-%d}"
                else:
                    period = "unspecified"

                # Scenario indicator strip
                if st.session_state.scenario == "optimised":
                    n_shifted = int(estimated["mode_shifted"].sum()) if "mode_shifted" in estimated.columns else 0
                    sc_col1, sc_col2 = st.columns([4, 1])
                    with sc_col1:
                        st.markdown(
                            f"<div class='headline-ok' style='margin-bottom:0.6rem'>"
                            f"<b>Optimised scenario active</b> &middot; {n_shifted} flight(s) shifted to a "
                            f"lower-CO2 mode &middot; saved {saving_potential:,.1f} t CO2 vs. as-planned."
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                    with sc_col2:
                        if st.button("Reset to as-planned", use_container_width=True, key="reset_scenario"):
                            st.session_state.scenario = "as_planned"
                            st.rerun()

                # Metadata strip
                n_unmatched = int(estimated["estimated_co2"].isna().sum())
                match_note = "" if n_unmatched == 0 else f" - {n_unmatched} trip(s) without route match"
                scenario_label = "as planned" if st.session_state.scenario == "as_planned" else "optimised (mode shift applied)"
                st.markdown(
                    f"<div class='meta-strip'>"
                    f"<span><b>Source</b> {src_label}</span>"
                    f"<span><b>Period</b> {period}</span>"
                    f"<span><b>Trips</b> {n_input:,}</span>"
                    f"<span><b>Scenario</b> {scenario_label}</span>"
                    f"<span><b>Reference</b> {hist_file.name} ({len(hist):,} historical trips)</span>"
                    f"<span><b>Method</b> route x mode average{match_note}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                # ------------------------------------------------------------------
                # Section 1: Overview KPIs
                # ------------------------------------------------------------------
                total_co2 = float(estimated["estimated_co2"].sum())
                total_budget = sum(budgets.values()) if budgets else 0
                n_trips = len(estimated)

                st.markdown("<div class='section-title'>Overview</div>", unsafe_allow_html=True)

                kc = st.columns(4)

                with kc[0]:
                    if total_budget > 0:
                        delta_t = total_co2 - total_budget
                        delta_pct = delta_t / total_budget * 100
                        if delta_t > 0:
                            delta_html = f"<span class='kpi-delta-bad'>+{delta_t:,.1f} t over budget ({delta_pct:+.0f}%)</span>"
                        else:
                            delta_html = f"<span class='kpi-delta-ok'>{delta_t:,.1f} t under budget ({delta_pct:+.0f}%)</span>"
                    else:
                        delta_html = "<span class='kpi-delta-neutral'>no budget loaded</span>"
                    st.markdown(
                        f"<div class='kpi-card'>"
                        f"<div class='kpi-label'>Total CO2 emissions</div>"
                        f"<div class='kpi-value'>{total_co2:,.1f} t</div>"
                        f"{delta_html}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                with kc[1]:
                    compliance = (total_co2 / total_budget * 100) if total_budget > 0 else 0
                    if compliance > 100:
                        comp_class = "kpi-delta-bad"
                    elif compliance > 85:
                        comp_class = "kpi-delta-neutral"
                    else:
                        comp_class = "kpi-delta-ok"
                    st.markdown(
                        f"<div class='kpi-card'>"
                        f"<div class='kpi-label'>Budget utilisation</div>"
                        f"<div class='kpi-value'>{compliance:.0f}%</div>"
                        f"<span class='{comp_class}'>of {total_budget:,.0f} t allocated</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                with kc[2]:
                    if st.session_state.scenario == "optimised":
                        pct_save = (saving_potential / (total_co2 + saving_potential) * 100) if (total_co2 + saving_potential) > 0 else 0
                        st.markdown(
                            f"<div class='kpi-card'>"
                            f"<div class='kpi-label'>CO2 saved by mode shift</div>"
                            f"<div class='kpi-value' style='color:{COLOR["ok"]}'>-{saving_potential:,.1f} t</div>"
                            f"<span class='kpi-delta-ok'>{pct_save:.0f}% lower than as-planned</span>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                    else:
                        pct_save = (saving_potential / total_co2 * 100) if total_co2 > 0 else 0
                        st.markdown(
                            f"<div class='kpi-card'>"
                            f"<div class='kpi-label'>Reduction potential</div>"
                            f"<div class='kpi-value'>{saving_potential:,.1f} t</div>"
                            f"<span class='kpi-delta-ok'>via mode shift ({pct_save:.0f}% of total)</span>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                with kc[3]:
                    avg_per_trip = (total_co2 / n_trips * 1000) if n_trips else 0
                    st.markdown(
                        f"<div class='kpi-card'>"
                        f"<div class='kpi-label'>Trips analysed</div>"
                        f"<div class='kpi-value'>{n_trips:,}</div>"
                        f"<span class='kpi-delta-neutral'>avg {avg_per_trip:,.0f} kg / trip</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                # Headline
                if total_budget > 0:
                    if total_co2 > total_budget:
                        st.markdown(
                            f"<div class='headline-bad' style='margin-top:1rem'>"
                            f"Total emissions exceed the combined CO2 budget by "
                            f"<b>{total_co2 - total_budget:,.1f} t</b> ({(total_co2/total_budget-1)*100:+.0f}%). "
                            f"See the BU breakdown below to identify where to act."
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                    elif total_co2 > 0.85 * total_budget:
                        st.markdown(
                            f"<div class='headline-warn' style='margin-top:1rem'>"
                            f"Emissions are within budget but approaching the limit "
                            f"({(total_co2/total_budget)*100:.0f}% utilised)."
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f"<div class='headline-ok' style='margin-top:1rem'>"
                            f"Emissions are well within the combined CO2 budget "
                            f"({(total_co2/total_budget)*100:.0f}% utilised)."
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                # ------------------------------------------------------------------
                # Section 2: BU performance
                # ------------------------------------------------------------------
                st.markdown("<div class='section-title'>Budget compliance by Business Unit</div>", unsafe_allow_html=True)
                st.markdown(
                    "<div class='section-hint'>Status gauges and ranking, side by side. "
                    "Line on each gauge marks the BU's annual budget.</div>",
                    unsafe_allow_html=True,
                )

                bu_emissions = estimated.groupby("business_unit")["estimated_co2"].sum().to_dict()
                bus_present = sorted(set(bu_emissions.keys()) | set(budgets.keys()))

                left, right = st.columns([3, 2])

                with left:
                    if bus_present:
                        rows = [bus_present[i:i+2] for i in range(0, len(bus_present), 2)]
                        for row in rows:
                            cols = st.columns(2)
                            for col, bu in zip(cols, row):
                                col.plotly_chart(
                                    gauge(
                                        bu_emissions.get(bu, 0),
                                        budgets.get(bu, np.nan),
                                        bu, BU_COLOR.get(bu, COLOR["ink"]),
                                    ),
                                    use_container_width=True,
                                    config={"displayModeBar": False},
                                )

                with right:
                    st.plotly_chart(
                        bar_bu_vs_budget(bu_emissions, budgets),
                        use_container_width=True,
                        config={"displayModeBar": False},
                    )

                # Per-BU headlines
                hl_cols = st.columns(len(bus_present)) if bus_present else []
                for col, bu in zip(hl_cols, bus_present):
                    actual = bu_emissions.get(bu, 0)
                    bud = budgets.get(bu, np.nan)
                    if pd.isna(bud) or bud <= 0:
                        col.markdown(
                            f"<div class='headline-warn'><b>{bu}</b>: no budget set "
                            f"(actual {actual:.1f} t)</div>",
                            unsafe_allow_html=True,
                        )
                    elif actual > bud:
                        col.markdown(
                            f"<div class='headline-bad'><b>{bu}</b> over by "
                            f"{actual - bud:,.1f} t ({(actual/bud - 1)*100:+.0f}%)</div>",
                            unsafe_allow_html=True,
                        )
                    elif actual > 0.85 * bud:
                        col.markdown(
                            f"<div class='headline-warn'><b>{bu}</b> approaching limit "
                            f"({(actual/bud)*100:.0f}% used)</div>",
                            unsafe_allow_html=True,
                        )
                    else:
                        col.markdown(
                            f"<div class='headline-ok'><b>{bu}</b> on track "
                            f"({(actual/bud)*100:.0f}% used)</div>",
                            unsafe_allow_html=True,
                        )

                # ------------------------------------------------------------------
                # Section 3: Geography
                # ------------------------------------------------------------------
                st.markdown("<div class='section-title'>Geographic distribution</div>", unsafe_allow_html=True)
                st.markdown(
                    "<div class='section-hint'>Line thickness scales with total CO2 on each route. "
                    "Colour encodes transport mode. Hover over a route for details.</div>",
                    unsafe_allow_html=True,
                )

                map_left, _ = st.columns([1, 4])
                with map_left:
                    map_region = st.selectbox(
                        "Region", list(REGION_VIEWS.keys()), index=0,
                        label_visibility="collapsed", key="map_region",
                    )

                route_summary = (
                    estimated.groupby(["departure_iata", "arrival_iata", "transport_mode"])
                    .agg(
                        total_co2=("estimated_co2", "sum"),
                        n_trips=("estimated_co2", "count"),
                        dep_lat=("departure_lat", "first"),
                        dep_lon=("departure_lon", "first"),
                        arr_lat=("arrival_lat", "first"),
                        arr_lon=("arrival_lon", "first"),
                    )
                    .reset_index()
                )
                st.plotly_chart(
                    world_map(route_summary, region=map_region),
                    use_container_width=True, config={"displayModeBar": False},
                )

                # ------------------------------------------------------------------
                # Section 4: Reduction levers
                # ------------------------------------------------------------------
                st.markdown("<div class='section-title'>Reduction levers</div>", unsafe_allow_html=True)
                st.markdown(
                    "<div class='section-hint'>For every flight in the input, the dashboard checks if a "
                    "lower-CO2 mode (train, bus, rental car) was historically used on the same route, "
                    "and lists the routes with the largest aggregate saving potential.</div>",
                    unsafe_allow_html=True,
                )

                if alts.empty:
                    st.markdown(
                        "<div class='headline-ok'>No greener alternatives found in the historical data "
                        "for the analysed flights.</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    summary = (
                        alts.groupby(["departure_iata", "arrival_iata", "alt_mode"])
                        .agg(
                            n_flights=("saving_t", "count"),
                            avg_flight_co2=("estimated_co2", "mean"),
                            avg_alt_co2=("alt_co2", "mean"),
                            total_saving_t=("saving_t", "sum"),
                            avg_saving_pct=("saving_pct", "mean"),
                        )
                        .reset_index()
                        .sort_values("total_saving_t", ascending=False)
                        .rename(columns={
                            "departure_iata":  "From",
                            "arrival_iata":    "To",
                            "alt_mode":        "Alternative",
                            "n_flights":       "Flights",
                            "avg_flight_co2":  "Avg flight (t)",
                            "avg_alt_co2":     "Avg alt. (t)",
                            "total_saving_t":  "Saving (t)",
                            "avg_saving_pct":  "Saving %",
                        })
                    )

                    sav_col1, sav_col2 = st.columns([1, 2])
                    with sav_col1:
                        st.markdown(
                            f"<div class='kpi-card'>"
                            f"<div class='kpi-label'>Total saving potential</div>"
                            f"<div class='kpi-value' style='color:{COLOR["ok"]}'>{saving_potential:,.1f} t</div>"
                            f"<span class='kpi-delta-neutral'>across {summary.shape[0]} route(s)</span>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                    with sav_col2:
                        st.dataframe(
                            summary.head(15).style.format({
                                "Avg flight (t)": "{:.3f}",
                                "Avg alt. (t)":   "{:.3f}",
                                "Saving (t)":     "{:.2f}",
                                "Saving %":       "{:.1f}%",
                            }),
                            use_container_width=True, hide_index=True,
                        )

                    btn_col1, btn_col2 = st.columns([1, 3])
                    with btn_col1:
                        if st.button(
                            "Apply alternatives",
                            type="primary", use_container_width=True, key="apply_alts",
                            help="Replace each flight that has a greener alternative with that "
                                 "alternative, then recompute the dashboard against the budget.",
                        ):
                            st.session_state.scenario = "optimised"
                            st.rerun()
                    with btn_col2:
                        st.markdown(
                            f"<div class='section-hint' style='margin-top:0.6rem'>"
                            f"Applies the {len(alts):,} suggested mode shifts above. "
                            f"Gauges, banner and KPIs will recompute against the same budgets."
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                # ------------------------------------------------------------------
                # Section 5: Detail (collapsed)
                # ------------------------------------------------------------------
                with st.expander("Detail data and export"):
                    cols = [
                        c for c in [
                            "date", "business_unit", "person_type", "transport_mode",
                            "departure_iata", "arrival_iata", "km", "estimated_co2",
                            "cost_CHF", "travel_purpose",
                        ] if c in estimated.columns
                    ]
                    st.dataframe(
                        estimated[cols].head(2000),
                        use_container_width=True, hide_index=True,
                    )
                    st.caption(f"Showing first 2000 of {len(estimated):,} rows.")

                    buf = BytesIO()
                    with pd.ExcelWriter(buf, engine="openpyxl") as w:
                        estimated.to_excel(w, sheet_name="estimated", index=False)
                        route_summary.to_excel(w, sheet_name="routes", index=False)
                        if not alts.empty:
                            summary.to_excel(w, sheet_name="alternatives", index=False)
                    buf.seek(0)
                    st.download_button(
                        "Download analysis as Excel",
                        data=buf.getvalue(),
                        file_name="co2_dashboard_export.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )