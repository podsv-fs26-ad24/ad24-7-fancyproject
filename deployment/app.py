
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import math
import numpy as np
import pandas as pd
import streamlit as st


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
APP_TITLE = "CO2 Travel Planning Dashboard"
APP_CAPTION = (
    "Upload planned future trips, compare estimated emissions against a "
    "provisional CO2 budget, and identify lower-emission alternatives."
)

DEFAULT_HISTORICAL_FILE = Path(__file__).with_name("traveldata-export.xlsx")
REQUIRED_UPLOAD_COLUMNS = ["departure_iata", "arrival_iata", "travel_purpose"]

# Conservative, explainable fallback factors in kg CO2e per passenger-km.
# These are only used when no historical route evidence exists.
MODE_FACTORS_KG_PER_KM = {
    "flight": 0.255,
    "train": 0.035,
    "bus": 0.082,
    "rental_car": 0.171,
}

# Heuristic feasibility thresholds for alternatives when the exact route has
# no historical alternative records.
TRAIN_MAX_KM = 900
BUS_MAX_KM = 350

MODE_LABELS = {
    "flight": "Flight",
    "train": "Train",
    "bus": "Bus",
    "rental_car": "Rental car",
}


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def normalize_iata(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.upper()
        .replace({"NAN": np.nan, "NONE": np.nan, "": np.nan})
    )


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(d_lambda / 2) ** 2
    )
    return 2 * radius * math.asin(math.sqrt(a))


def format_kg(value: float) -> str:
    return f"{value:,.0f} kg".replace(",", "'")


def route_key(dep: str, arr: str) -> str:
    return f"{dep}-{arr}"


def undirected_route_key(dep: str, arr: str) -> str:
    return "-".join(sorted([dep, arr]))


# -----------------------------------------------------------------------------
# Data preparation
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_historical_data(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="travel_data")

    expected_cols = {
        "transport_mode",
        "departure_iata",
        "arrival_iata",
        "travel_purpose",
        "CO2e RFI2.7 (t)",
        "km",
        "departure_lat",
        "departure_lon",
        "arrival_lat",
        "arrival_lon",
        "train_alternative_available",
    }
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Historical file is missing required columns: {sorted(missing)}"
        )

    df = df.copy()
    df["departure_iata"] = normalize_iata(df["departure_iata"])
    df["arrival_iata"] = normalize_iata(df["arrival_iata"])
    df["transport_mode"] = (
        df["transport_mode"].astype(str).str.strip().str.lower()
    )
    df["travel_purpose"] = (
        df["travel_purpose"].astype(str).str.strip().str.lower()
    )
    df["co2_kg"] = pd.to_numeric(df["CO2e RFI2.7 (t)"], errors="coerce") * 1000
    df["km"] = pd.to_numeric(df["km"], errors="coerce")
    df["route"] = df["departure_iata"] + "-" + df["arrival_iata"]
    df["route_undirected"] = df.apply(
        lambda x: undirected_route_key(x["departure_iata"], x["arrival_iata"]), axis=1
    )

    # Robust boolean normalisation
    df["train_alternative_available"] = (
        df["train_alternative_available"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False})
    )

    df = df.dropna(subset=["departure_iata", "arrival_iata", "transport_mode"])
    return df


@st.cache_data(show_spinner=False)
def build_airport_reference(hist: pd.DataFrame) -> pd.DataFrame:
    dep = hist[["departure_iata", "departure_lat", "departure_lon"]].rename(
        columns={
            "departure_iata": "iata",
            "departure_lat": "lat",
            "departure_lon": "lon",
        }
    )
    arr = hist[["arrival_iata", "arrival_lat", "arrival_lon"]].rename(
        columns={
            "arrival_iata": "iata",
            "arrival_lat": "lat",
            "arrival_lon": "lon",
        }
    )
    airports = pd.concat([dep, arr], ignore_index=True)
    airports["lat"] = pd.to_numeric(airports["lat"], errors="coerce")
    airports["lon"] = pd.to_numeric(airports["lon"], errors="coerce")
    airports = airports.dropna(subset=["iata", "lat", "lon"]).drop_duplicates("iata")
    return airports


@st.cache_data(show_spinner=False)
def build_route_statistics(hist: pd.DataFrame) -> dict[str, pd.DataFrame]:
    base = (
        hist.dropna(subset=["co2_kg", "km"])
        .groupby(["route", "route_undirected", "travel_purpose", "transport_mode"], dropna=False)
        .agg(
            trips=("transport_mode", "size"),
            median_co2_kg=("co2_kg", "median"),
            mean_co2_kg=("co2_kg", "mean"),
            median_km=("km", "median"),
        )
        .reset_index()
    )

    route_mode = (
        hist.dropna(subset=["co2_kg", "km"])
        .groupby(["route", "route_undirected", "transport_mode"], dropna=False)
        .agg(
            trips=("transport_mode", "size"),
            median_co2_kg=("co2_kg", "median"),
            mean_co2_kg=("co2_kg", "mean"),
            median_km=("km", "median"),
        )
        .reset_index()
    )

    purpose_mode = (
        hist.dropna(subset=["co2_kg", "km"])
        .groupby(["travel_purpose", "transport_mode"], dropna=False)
        .agg(
            trips=("transport_mode", "size"),
            median_co2_kg=("co2_kg", "median"),
            mean_co2_kg=("co2_kg", "mean"),
            median_km=("km", "median"),
        )
        .reset_index()
    )

    route_dominant = (
        hist.groupby(["route", "route_undirected", "transport_mode"])
        .size()
        .reset_index(name="trips")
        .sort_values(["route", "trips"], ascending=[True, False])
        .drop_duplicates("route")
        .rename(columns={"transport_mode": "dominant_mode"})
    )

    train_route_flags = (
        hist.groupby(["route", "route_undirected"], dropna=False)["train_alternative_available"]
        .agg(lambda s: bool(pd.Series(s).fillna(False).any()))
        .reset_index()
        .rename(columns={"train_alternative_available": "train_alt_seen"})
    )

    return {
        "route_purpose_mode": base,
        "route_mode": route_mode,
        "purpose_mode": purpose_mode,
        "route_dominant": route_dominant,
        "train_route_flags": train_route_flags,
    }


# -----------------------------------------------------------------------------
# Estimation logic
# -----------------------------------------------------------------------------
@dataclass
class TripEstimate:
    reference_mode: str
    reference_co2_kg: float
    reference_distance_km: float
    reference_source: str
    recommended_mode: str
    recommended_co2_kg: float
    recommended_source: str
    saving_kg: float
    saving_pct: float


def lookup_distance_km(dep: str, arr: str, airports: pd.DataFrame) -> Optional[float]:
    dep_row = airports.loc[airports["iata"] == dep]
    arr_row = airports.loc[airports["iata"] == arr]
    if dep_row.empty or arr_row.empty:
        return None
    return haversine_km(
        float(dep_row["lat"].iloc[0]),
        float(dep_row["lon"].iloc[0]),
        float(arr_row["lat"].iloc[0]),
        float(arr_row["lon"].iloc[0]),
    )


def fallback_emissions(distance_km: float, mode: str) -> float:
    factor = MODE_FACTORS_KG_PER_KM[mode]
    return distance_km * factor


def choose_reference_mode(
    dep: str,
    arr: str,
    purpose: str,
    stats: dict[str, pd.DataFrame],
    distance_km: Optional[float],
) -> tuple[str, float, float, str]:
    rk = route_key(dep, arr)
    urk = undirected_route_key(dep, arr)

    rpm = stats["route_purpose_mode"]
    route_mode = stats["route_mode"]
    dominant = stats["route_dominant"]

    # 1) Exact route + purpose + dominant record by trip count
    subset = rpm[(rpm["route"] == rk) & (rpm["travel_purpose"] == purpose)]
    if not subset.empty:
        best = subset.sort_values("trips", ascending=False).iloc[0]
        return (
            str(best["transport_mode"]),
            float(best["median_co2_kg"]),
            float(best["median_km"]),
            "Historical exact route + purpose",
        )

    # 2) Exact route dominant mode
    dom = dominant[dominant["route"] == rk]
    if not dom.empty:
        ref_mode = str(dom["dominant_mode"].iloc[0])
        rm = route_mode[(route_mode["route"] == rk) & (route_mode["transport_mode"] == ref_mode)]
        if not rm.empty:
            row = rm.iloc[0]
            return (
                ref_mode,
                float(row["median_co2_kg"]),
                float(row["median_km"]),
                "Historical exact route",
            )

    # 3) Undirected route dominant mode
    dom = dominant[dominant["route_undirected"] == urk]
    if not dom.empty:
        ref_mode = str(dom["dominant_mode"].iloc[0])
        rm = route_mode[
            (route_mode["route_undirected"] == urk) & (route_mode["transport_mode"] == ref_mode)
        ]
        if not rm.empty:
            row = rm.sort_values("trips", ascending=False).iloc[0]
            return (
                ref_mode,
                float(row["median_co2_kg"]),
                float(row["median_km"]),
                "Historical reverse/similar route",
            )

    # 4) Fallback to flight for unknown routes
    if distance_km is None:
        distance_km = 1000.0
    return (
        "flight",
        float(fallback_emissions(distance_km, "flight")),
        float(distance_km),
        "Distance-based fallback",
    )


def choose_recommended_mode(
    dep: str,
    arr: str,
    purpose: str,
    distance_km: float,
    stats: dict[str, pd.DataFrame],
) -> tuple[str, float, str]:
    rk = route_key(dep, arr)
    urk = undirected_route_key(dep, arr)

    rpm = stats["route_purpose_mode"]
    route_mode = stats["route_mode"]
    purpose_mode = stats["purpose_mode"]
    train_flags = stats["train_route_flags"]

    candidates: list[tuple[str, float, int, str]] = []

    def add_candidates(df: pd.DataFrame, source: str) -> None:
        if df.empty:
            return
        for _, row in df.iterrows():
            candidates.append((
                str(row["transport_mode"]),
                float(row["median_co2_kg"]),
                int(row["trips"]),
                source,
            ))

    add_candidates(
        rpm[(rpm["route"] == rk) & (rpm["travel_purpose"] == purpose)],
        "Historical exact route + purpose",
    )
    add_candidates(
        route_mode[route_mode["route"] == rk],
        "Historical exact route",
    )
    add_candidates(
        route_mode[route_mode["route_undirected"] == urk],
        "Historical reverse/similar route",
    )
    add_candidates(
        purpose_mode[purpose_mode["travel_purpose"] == purpose],
        "Historical purpose-level pattern",
    )

    # Heuristic alternatives if no explicit route-level candidate exists.
    has_train_route_flag = False
    train_flag_row = train_flags[train_flags["route_undirected"] == urk]
    if not train_flag_row.empty:
        has_train_route_flag = bool(train_flag_row["train_alt_seen"].iloc[0])

    if has_train_route_flag or distance_km <= TRAIN_MAX_KM:
        candidates.append((
            "train",
            fallback_emissions(distance_km, "train"),
            1,
            "Distance-based heuristic",
        ))

    if distance_km <= BUS_MAX_KM:
        candidates.append((
            "bus",
            fallback_emissions(distance_km, "bus"),
            1,
            "Distance-based heuristic",
        ))

    # Always include flight fallback so there is at least one option.
    candidates.append((
        "flight",
        fallback_emissions(distance_km, "flight"),
        1,
        "Distance-based fallback",
    ))

    # Pick the lowest-emission candidate, tie-break with more evidence.
    best = sorted(candidates, key=lambda x: (x[1], -x[2]))[0]
    return best[0], best[1], best[3]


def estimate_trip(
    dep: str,
    arr: str,
    purpose: str,
    airports: pd.DataFrame,
    stats: dict[str, pd.DataFrame],
) -> TripEstimate:
    purpose = str(purpose).strip().lower()
    distance_km = lookup_distance_km(dep, arr, airports)

    ref_mode, ref_co2, ref_dist, ref_source = choose_reference_mode(
        dep=dep,
        arr=arr,
        purpose=purpose,
        stats=stats,
        distance_km=distance_km,
    )

    if distance_km is None:
        distance_km = ref_dist

    rec_mode, rec_co2, rec_source = choose_recommended_mode(
        dep=dep,
        arr=arr,
        purpose=purpose,
        distance_km=distance_km,
        stats=stats,
    )

    saving_kg = max(ref_co2 - rec_co2, 0.0)
    saving_pct = (saving_kg / ref_co2 * 100.0) if ref_co2 > 0 else 0.0

    return TripEstimate(
        reference_mode=ref_mode,
        reference_co2_kg=ref_co2,
        reference_distance_km=distance_km,
        reference_source=ref_source,
        recommended_mode=rec_mode,
        recommended_co2_kg=rec_co2,
        recommended_source=rec_source,
        saving_kg=saving_kg,
        saving_pct=saving_pct,
    )


def validate_upload(df: pd.DataFrame) -> list[str]:
    missing_cols = [c for c in REQUIRED_UPLOAD_COLUMNS if c not in df.columns]
    errors = []
    if missing_cols:
        errors.append(
            "Missing required columns: " + ", ".join(missing_cols)
        )
        return errors

    cleaned = df.copy()
    cleaned["departure_iata"] = normalize_iata(cleaned["departure_iata"])
    cleaned["arrival_iata"] = normalize_iata(cleaned["arrival_iata"])
    cleaned["travel_purpose"] = (
        cleaned["travel_purpose"].astype(str).str.strip().str.lower()
    )

    if cleaned["departure_iata"].isna().any():
        errors.append("Some rows have missing departure_iata.")
    if cleaned["arrival_iata"].isna().any():
        errors.append("Some rows have missing arrival_iata.")
    if cleaned["travel_purpose"].isna().any():
        errors.append("Some rows have missing travel_purpose.")

    same_airport = cleaned["departure_iata"] == cleaned["arrival_iata"]
    if same_airport.any():
        errors.append("Some rows use the same airport for departure and arrival.")

    return errors


def estimate_uploaded_trips(
    upload_df: pd.DataFrame,
    airports: pd.DataFrame,
    stats: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    df = upload_df.copy()
    df["departure_iata"] = normalize_iata(df["departure_iata"])
    df["arrival_iata"] = normalize_iata(df["arrival_iata"])
    df["travel_purpose"] = (
        df["travel_purpose"].astype(str).str.strip().str.lower()
    )

    estimates = [
        estimate_trip(
            dep=row["departure_iata"],
            arr=row["arrival_iata"],
            purpose=row["travel_purpose"],
            airports=airports,
            stats=stats,
        )
        for _, row in df.iterrows()
    ]

    est_df = pd.DataFrame([e.__dict__ for e in estimates])
    out = pd.concat([df.reset_index(drop=True), est_df], axis=1)

    out["route"] = out["departure_iata"] + "-" + out["arrival_iata"]
    out["status"] = np.where(
        out["recommended_co2_kg"] < out["reference_co2_kg"],
        "Alternative available",
        "No better alternative found",
    )
    out["switch_recommended"] = np.where(
        out["recommended_mode"] != out["reference_mode"],
        "Yes",
        "No",
    )
    return out


# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption(APP_CAPTION)

with st.expander("How this MVP works", expanded=False):
    st.markdown(
        """
        - Historical travel data is used as the reference base.
        - The uploaded Excel file should contain at least: `departure_iata`, `arrival_iata`, `travel_purpose`.
        - For each planned trip, the app estimates a **reference emission** from historical evidence.
        - It then checks whether a lower-emission **recommended mode** exists, especially train or bus.
        - When no exact historical route is found, the app falls back to a distance-based estimate.
        """
    )

# Sidebar controls
st.sidebar.header("Settings")
historical_path = st.sidebar.text_input(
    "Historical reference workbook path",
    value=str(DEFAULT_HISTORICAL_FILE),
    help="The workbook must contain a 'travel_data' sheet.",
)
budget_kg = st.sidebar.number_input(
    "Provisional CO2 budget (kg)",
    min_value=0.0,
    value=1000.0,
    step=100.0,
    help="This budget is compared against the sum of reference emissions for the uploaded planned trips.",
)

try:
    hist = load_historical_data(historical_path)
    airports = build_airport_reference(hist)
    stats = build_route_statistics(hist)
except Exception as exc:
    st.error(f"Could not load the historical workbook: {exc}")
    st.stop()

col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Historical trips", f"{len(hist):,}".replace(",", "'"))
col_b.metric("Known routes", f"{hist['route'].nunique():,}".replace(",", "'"))
col_c.metric("Known airports", f"{len(airports):,}".replace(",", "'"))
col_d.metric("Transport modes", ", ".join(sorted(hist["transport_mode"].dropna().unique())))

uploaded_file = st.file_uploader(
    "Upload future trip requests (.xlsx)",
    type=["xlsx"],
    help="Required columns: departure_iata, arrival_iata, travel_purpose",
)

example_df = pd.DataFrame(
    {
        "departure_iata": ["AMS", "BRU", "ZRH", "IST", "TXL"],
        "arrival_iata": ["ZRH", "ZRH", "MUC", "ZRH", "ZRH"],
        "travel_purpose": [
            "internal_meeting",
            "workshop",
            "training",
            "internal_meeting",
            "internal_meeting",
        ],
    }
)

with st.expander("Expected upload structure", expanded=False):
    st.dataframe(example_df, use_container_width=True)

if uploaded_file is None:
    st.info("Upload a future-trip Excel file to run the analysis.")
    st.stop()

try:
    planned = pd.read_excel(uploaded_file)
except Exception as exc:
    st.error(f"Could not read the uploaded file: {exc}")
    st.stop()

errors = validate_upload(planned)
if errors:
    for msg in errors:
        st.error(msg)
    st.stop()

results = estimate_uploaded_trips(planned, airports, stats)

reference_total = float(results["reference_co2_kg"].sum())
recommended_total = float(results["recommended_co2_kg"].sum())
saving_total = float(results["saving_kg"].sum())
budget_gap = budget_kg - reference_total
within_budget = reference_total <= budget_kg

st.subheader("Budget overview")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Budget", format_kg(budget_kg))
k2.metric("Estimated current plan", format_kg(reference_total))
k3.metric("Best-case with alternatives", format_kg(recommended_total))
k4.metric("Potential saving", format_kg(saving_total))

if within_budget:
    st.success(
        f"The uploaded trip plan stays within budget by {format_kg(abs(budget_gap))}."
    )
else:
    st.warning(
        f"The uploaded trip plan exceeds the budget by {format_kg(abs(budget_gap))}."
    )

st.subheader("Mode mix")
mix_col1, mix_col2 = st.columns(2)

with mix_col1:
    ref_mix = (
        results["reference_mode"]
        .map(MODE_LABELS)
        .value_counts()
        .rename_axis("Mode")
        .reset_index(name="Trips")
    )
    st.markdown("**Reference plan**")
    st.dataframe(ref_mix, use_container_width=True, hide_index=True)

with mix_col2:
    rec_mix = (
        results["recommended_mode"]
        .map(MODE_LABELS)
        .value_counts()
        .rename_axis("Mode")
        .reset_index(name="Trips")
    )
    st.markdown("**Recommended plan**")
    st.dataframe(rec_mix, use_container_width=True, hide_index=True)

st.subheader("Trip-level analysis")
display_cols = [
    "route",
    "travel_purpose",
    "reference_mode",
    "reference_distance_km",
    "reference_co2_kg",
    "recommended_mode",
    "recommended_co2_kg",
    "saving_kg",
    "saving_pct",
    "status",
    "switch_recommended",
    "reference_source",
    "recommended_source",
]
display_df = results[display_cols].copy()
display_df["reference_distance_km"] = display_df["reference_distance_km"].round(0)
display_df["reference_co2_kg"] = display_df["reference_co2_kg"].round(1)
display_df["recommended_co2_kg"] = display_df["recommended_co2_kg"].round(1)
display_df["saving_kg"] = display_df["saving_kg"].round(1)
display_df["saving_pct"] = display_df["saving_pct"].round(1)
st.dataframe(display_df, use_container_width=True, hide_index=True)

st.subheader("Priority switches")
priority = (
    results.loc[results["switch_recommended"] == "Yes"]
    .sort_values("saving_kg", ascending=False)
    .loc[:, [
        "route",
        "travel_purpose",
        "reference_mode",
        "recommended_mode",
        "reference_co2_kg",
        "recommended_co2_kg",
        "saving_kg",
        "saving_pct",
    ]]
    .copy()
)

if priority.empty:
    st.info("No lower-emission mode switch was identified for the uploaded trips.")
else:
    priority["reference_co2_kg"] = priority["reference_co2_kg"].round(1)
    priority["recommended_co2_kg"] = priority["recommended_co2_kg"].round(1)
    priority["saving_kg"] = priority["saving_kg"].round(1)
    priority["saving_pct"] = priority["saving_pct"].round(1)
    st.dataframe(priority, use_container_width=True, hide_index=True)

csv_data = results.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download results as CSV",
    data=csv_data,
    file_name="trip_budget_analysis.csv",
    mime="text/csv",
)

st.subheader("Important assumptions and current limits")
st.markdown(
    """
    1. This version uses the historical workbook as a reference model and not as a hard budget source.
    2. When an exact route is unknown, the app falls back to airport-distance estimates.
    3. Train and bus recommendations are based on historical evidence plus distance heuristics.
    4. This MVP does **not** yet query live rail/bus APIs for real schedules or connections.
    5. The budget comparison is currently done against the estimated **reference plan** for the uploaded trips.
    """
)
