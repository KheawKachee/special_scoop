from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# =======================
# Load data
# =======================
root = Path.cwd()
while not (root / ".git").exists() and root.parent != root:
    root = root.parent

DATA_PATH = root / "data" / "fbref"
year = "25_26"
prefix = "one"

METRIC_PRESETS = {
    "Custom": None,
    "Progressive 1": [
        "npxg_xg_assist_per90%",
        "passes_completed_per90%",
        "passes_into_final_third_per90%",
        "progressive_passes_per90%",
        "gca_per90%",
        "carries_distance_per90%",
        "progressive_carries_per90%",
        "ball_recoveries_per90%",
    ],
    "Creation Action 1": ["xg_assist_per90%", "gca_per90%", "sca_per90%", "passes_into_penalty_area_per90%", "assisted_shots_per90%"],
    "Progressive_2": ["progressive_passes_per90%", "passes_progressive_distance_per90%", "progressive_carries_per90%", "carries_progressive_distance_per90%", "touches_att_3rd_per90%"],
    "Defensive 1": ["tackles_interceptions_per90%", "ball_recoveries_per90%", "interceptions_per90%", "blocks_per90%", "clearances_per90%"],
}

raw_df = pd.read_json(DATA_PATH / "PL_outfield" / year / f"PL_outfield_{year}_{prefix}.json")

# Remove duplicate columns (keep first occurrence)
raw_df = raw_df.loc[:, ~raw_df.columns.duplicated()]


def normalize_positions(df):
    """
    Works when position has already been one-hot encoded.
    Creates:
    - primary_position (DF / MF / FW / GK)
    """

    position_cols = [c for c in df.columns if c.startswith("position_")]

    if not position_cols:
        raise ValueError("No position_* columns found. Check preprocessing.")

    # Extract position labels
    pos_labels = [c.replace("position_", "") for c in position_cols]

    # Primary position = first active one (stable + deterministic)
    df["primary_position"] = df[position_cols].idxmax(axis=1).str.replace("position_", "", regex=False)

    return df


def player_per90_percentiles(df, per90=True, percentile=True):
    semantic_cols = {"ranker", "player", "nationality", "team", "age", "birth_year", "games", "games_starts", "minutes", "minutes_90s"}

    stats_cols = df.select_dtypes(include="number").columns
    cols_to_transform = [c for c in stats_cols if c not in semantic_cols]

    res_df = df[cols_to_transform].copy()

    # ---- Per 90 ----
    if per90:
        # Separate columns into those that need per90 and those that already have it
        base_cols = [c for c in res_df.columns if not c.endswith("_per90")]
        already_per90_cols = [c for c in res_df.columns if c.endswith("_per90")]

        # Only process base columns
        if base_cols:
            # Create new per90 dataframe with transformed base columns
            per90_transformed = res_df[base_cols].div(df["minutes_90s"], axis=0)
            per90_transformed.columns = [f"{c}_per90" for c in base_cols]

            # Keep already per90 columns as is
            already_per90_df = res_df[already_per90_cols]

            # Combine them
            res_df = pd.concat([per90_transformed, already_per90_df], axis=1)
        # If all columns already have _per90, keep res_df as is

    # ---- Percentiles (within cohort) ----
    if percentile:
        # Apply percentile ranking to all columns
        res_df = res_df.rank(pct=True)
        # Add % suffix
        res_df.columns = [f"{c}%" for c in res_df.columns]

    final_df = pd.concat([df[["player", "team"]], res_df], axis=1)

    return final_df


def plot_plotly_radar_compare(df, players, metrics, show_avg=False):
    fig = go.Figure()

    labels = metrics + [metrics[0]]

    for player in players:
        row = df[df["player"] == player].iloc[0]
        values = [row[m] * 100 for m in metrics]

        fig.add_trace(go.Scatterpolar(r=values + [values[0]], theta=labels, fill="toself", name=player, opacity=0.6, line=dict(width=1.5)))

    if show_avg:
        avg = df[metrics].mean().values * 100
        fig.add_trace(go.Scatterpolar(r=list(avg) + [avg[0]], theta=labels, name="Cohort Avg", line=dict(color="white", dash="dot", width=1.2), fill=None))

    fig.update_layout(
        template="plotly_dark",
        height=800,
        margin=dict(l=80, r=80, t=60, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5, font=dict(size=20)),
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(range=[0, 100], tickfont=dict(size=11), ticks="", showline=False, gridcolor="rgba(255,255,255,0.15)"),
            angularaxis=dict(tickfont=dict(size=14, color="white", family="Arial Black"), rotation=90, direction="clockwise", gridcolor="rgba(255,255,255,0.15)"),
        ),
    )

    return fig


# =======================
# Streamlit UI
# =======================
st.set_page_config(layout="wide")
st.title("⚽ Player Radar Comparison")

with st.sidebar:
    st.header("Controls")

    position = st.selectbox("Position cohort", ["DF", "MF", "FW", "GK"], index=2)

    show_avg = st.checkbox("Show average line", value=True)

# ---- Compute data ----
position_col = f"position_{position}"

if position_col not in raw_df.columns:
    st.error(f"Missing column: {position_col}")
    st.stop()

filtered_df = raw_df[raw_df[position_col] == 1].copy()
df = player_per90_percentiles(filtered_df, per90=True, percentile=True)

# Remove any duplicate columns that might have been created
df = df.loc[:, ~df.columns.duplicated()]

# ---- Player & metric selection ----
players = st.multiselect("Players", options=df["player"].unique(), default=list(df["player"].unique()[:2]))

metric_cols = [c for c in df.columns if c not in {"player", "team"}]

# Initialize session state
if "selected_metrics" not in st.session_state:
    st.session_state.selected_metrics = metric_cols[:6] if len(metric_cols) >= 6 else metric_cols

if "current_preset" not in st.session_state:
    st.session_state.current_preset = "Custom"


# Callback function for preset change
def on_preset_change():
    preset = st.session_state.preset_selector
    if preset != "Custom" and METRIC_PRESETS[preset]:
        # Filter to only include metrics that exist in the data
        preset_metrics = [m for m in METRIC_PRESETS[preset] if m in metric_cols]
        if preset_metrics:
            st.session_state.selected_metrics = preset_metrics
            st.session_state.current_preset = preset


# Preset selection with callback
preset = st.selectbox("Metric Preset", options=list(METRIC_PRESETS.keys()), index=list(METRIC_PRESETS.keys()).index(st.session_state.current_preset), key="preset_selector", on_change=on_preset_change)

# Metrics multiselect
metrics = st.multiselect("Metrics", options=metric_cols, default=st.session_state.selected_metrics)

if metrics != METRIC_PRESETS[preset]:
    st.session_state.current_preset = "Custom"

if len(players) < 2:
    st.info("Select at least two players.")
elif not metrics:
    st.info("Select at least one metric.")
else:
    fig = plot_plotly_radar_compare(df, players, metrics, show_avg=show_avg)

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Player Info")

    info_cols = ["player", "nationality", "team", "age", "minutes_90s", "position_GK", "position_DF", "position_MF", "position_FW"]

    info_table = raw_df[raw_df["player"].isin(players)].set_index("player")[info_cols[1:]]

    st.dataframe(info_table, use_container_width=True)

    st.subheader("Selected Metrics – Percentiles")

    table = df[df["player"].isin(players)].set_index("player")[metrics].mul(100).round(1)

    table = table.replace([float("inf"), float("-inf")], pd.NA)
    table = table.astype("float64")

    # Final safety check for duplicates before display
    table = table.loc[:, ~table.columns.duplicated()]

    st.dataframe(table, use_container_width=True)
