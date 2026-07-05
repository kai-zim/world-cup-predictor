"""Streamlit dashboard.

Reads exclusively from outputs/predictions and outputs/simulations -- it
never recomputes features or retrains a model itself. Run the CLI first:

    worldcup-predictor update-after-round --fixtures

MVP scope: 3 of the 8 planned pages (Champion Probabilities, Match
Predictor, Tournament Bracket). Team Comparison, Prediction Timeline,
Feature Importance, Simulation Explorer and Model Diagnostics are Phase 2
(see README roadmap) -- they need training/backtest artifacts this MVP
doesn't persist yet.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from worldcup_predictor.reporting.plots import (
    champion_probability_bar,
    match_outcome_comparison,
    stage_progression_chart,
)
from worldcup_predictor.simulation.tournament import STAGE_PROGRESSION
from worldcup_predictor.utils.config import get_settings

st.set_page_config(page_title="World Cup 2026 Knockout Predictor", layout="wide")

settings = get_settings()
PREDICTIONS_PATH = settings.worldcup_output_dir / "predictions" / "match_predictions.csv"
SIMULATION_PATH = settings.worldcup_output_dir / "simulations" / "tournament_probabilities.csv"


@st.cache_data
def load_predictions(path: str) -> pd.DataFrame | None:
    if not PREDICTIONS_PATH.exists():
        return None
    return pd.read_csv(PREDICTIONS_PATH, parse_dates=["date"])


@st.cache_data
def load_simulation(path: str) -> pd.DataFrame | None:
    if not SIMULATION_PATH.exists():
        return None
    return pd.read_csv(SIMULATION_PATH)


predictions = load_predictions(str(PREDICTIONS_PATH))
simulation = load_simulation(str(SIMULATION_PATH))

st.title("FIFA World Cup 2026 -- Knockout Stage Predictor")

if predictions is None or simulation is None:
    st.warning(
        "No outputs found yet. From the project root, run:\n\n"
        "```\nworldcup-predictor update-after-round --fixtures\n```\n\n"
        "(drop `--fixtures` once you have downloaded the real dataset), then reload this page."
    )
    st.stop()

page = st.sidebar.radio("Page", ["Champion Probabilities", "Match Predictor", "Tournament Bracket"])

if page == "Champion Probabilities":
    st.header("Champion Probabilities")
    n_sims = simulation.attrs.get("n_simulations", "100,000+")
    st.caption(f"Based on {n_sims} Monte Carlo tournament simulations.")
    st.plotly_chart(champion_probability_bar(simulation), use_container_width=True)
    st.plotly_chart(stage_progression_chart(simulation, STAGE_PROGRESSION), use_container_width=True)
    st.dataframe(simulation, use_container_width=True, hide_index=True)

elif page == "Match Predictor":
    st.header("Match Predictor")
    labelled = predictions.assign(
        label=predictions["home_team"] + " vs " + predictions["away_team"] + " (" + predictions["stage"] + ")"
    )
    choice = st.selectbox("Match", labelled["label"])
    row = labelled[labelled["label"] == choice].iloc[0]

    col1, col2 = st.columns([2, 1])
    with col1:
        st.plotly_chart(match_outcome_comparison(row), use_container_width=True)
    with col2:
        st.metric(
            "Expected goals (goal model)",
            f"{row['goal_model_expected_home_goals']:.2f} - {row['goal_model_expected_away_goals']:.2f}",
        )
        if pd.notna(row["winner"]):
            score = f"{int(row['home_score'])} - {int(row['away_score'])}"
            st.success(f"Played: {row['home_team']} {score} {row['away_team']}")
        else:
            st.info("Not played yet -- probabilities above are the model's prediction.")

else:  # Tournament Bracket
    st.header("Tournament Bracket")
    for stage in ["group_stage", *STAGE_PROGRESSION, "third_place"]:
        stage_matches = predictions[predictions["stage"] == stage]
        if stage_matches.empty:
            continue
        st.subheader(stage.replace("_", " ").title())
        for _, m in stage_matches.iterrows():
            if pd.notna(m["winner"]):
                score = f"{int(m['home_score'])} - {int(m['away_score'])}"
                st.write(f"{m['home_team']} **{score}** {m['away_team']}")
            else:
                st.write(
                    f"{m['home_team']} vs {m['away_team']} -- "
                    f"P(home) {m['outcome_model_p_home']:.0%} · "
                    f"P(draw) {m['outcome_model_p_draw']:.0%} · "
                    f"P(away) {m['outcome_model_p_away']:.0%}"
                )
