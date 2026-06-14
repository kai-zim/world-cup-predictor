"""Streamlit dashboard for the WC2026 predictor.

Run with:  streamlit run app/dashboard.py

Tabs implemented:
    * Champion Probabilities — bar chart of title odds
    * Match Predictor — single-fixture W/D/L + expected goals
    * Simulation Explorer — re-run Monte-Carlo with adjustable N

Tabs scaffolded (need richer data / artefacts):
    * Tournament Bracket, Prediction Timeline, Team Comparison,
      Feature Importance — marked with TODO and a placeholder message.

The dashboard uses the synthetic demo data by default so it runs out-of-the-box.
Point ``DATA_PATH`` at a real normalised results CSV for live forecasts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from wc2026.config.schema import AppConfig  # noqa: E402
from wc2026.models.elo import EloEngine  # noqa: E402
from wc2026.models.goal_model import (  # noqa: E402
    EloPoissonModel,
    outcome_probabilities,
    scoreline_matrix,
)
from wc2026.pipeline import run_forecast  # noqa: E402

st.set_page_config(page_title="WC2026 Predictor", layout="wide")


@st.cache_data
def _demo_history() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    teams = [f"Team{i:02d}" for i in range(32)]
    dates = pd.date_range("2022-12-19", periods=500, freq="3D")
    rows = []
    for d in dates:
        a, b = rng.choice(teams, 2, replace=False)
        sa, sb = int(a[4:]), int(b[4:])
        rows.append(
            dict(
                date=d, tournament="Friendly", home_team=a, away_team=b,
                home_score=int(rng.poisson(max(0.3, 2.0 - sa * 0.05))),
                away_score=int(rng.poisson(max(0.3, 2.0 - sb * 0.05))),
                neutral=True,
            )
        )
    return pd.DataFrame(rows)


def main() -> None:
    st.title("🏆 World Cup 2026 — Knockout Predictor")
    st.caption(
        "Elo → Poisson → Monte-Carlo. Demo runs on synthetic data; "
        "swap in a real results CSV for live forecasts."
    )

    cfg = AppConfig()
    history = _demo_history()
    teams = sorted(set(history["home_team"]) | set(history["away_team"]))

    n_sims = st.sidebar.slider("Monte-Carlo simulations", 1000, 100000, 10000, step=1000)
    cfg.simulation.n_simulations = n_sims

    engine = EloEngine(cfg.elo)
    engine.replay(history)
    ratings = engine.ratings_snapshot()

    tab_champ, tab_match, tab_sim, tab_todo = st.tabs(
        ["Champion Probabilities", "Match Predictor", "Simulation Explorer", "More (WIP)"]
    )

    with tab_champ:
        forecast, _r, _b = run_forecast(history, teams[:32], cfg)
        top = forecast.head(16)
        fig = px.bar(top, x="p_champion", y="team", orientation="h", title="Title probability (top 16)")
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(forecast, use_container_width=True)

    with tab_match:
        c1, c2 = st.columns(2)
        home = c1.selectbox("Home / Team A", teams, index=0)
        away = c2.selectbox("Away / Team B", teams, index=1)
        model = EloPoissonModel()
        lam_h, lam_a = model.expected_goals(ratings[home], ratings[away], neutral=True)
        mat = scoreline_matrix(lam_h, lam_a, cfg.poisson.max_goals)
        p_h, p_d, p_a = outcome_probabilities(mat)
        m1, m2, m3 = st.columns(3)
        m1.metric(f"{home} win", f"{p_h:.1%}")
        m2.metric("Draw", f"{p_d:.1%}")
        m3.metric(f"{away} win", f"{p_a:.1%}")
        st.write(f"Expected goals — {home}: **{lam_h:.2f}**, {away}: **{lam_a:.2f}**")

    with tab_sim:
        st.write(f"Current setting: **{n_sims:,}** simulations (adjust in sidebar).")
        if st.button("Run simulation"):
            forecast, _r, _b = run_forecast(history, teams[:32], cfg)
            st.dataframe(forecast.head(20), use_container_width=True)

    with tab_todo:
        st.info(
            "TODO(dashboard): Tournament Bracket, Prediction Timeline, "
            "Team Comparison and Feature Importance tabs require persisted "
            "bracket state, a forecast history log, and a trained learned "
            "model with SHAP values respectively."
        )


if __name__ == "__main__":
    main()
