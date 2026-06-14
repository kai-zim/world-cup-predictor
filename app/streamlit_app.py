import streamlit as st

from world_cup_predictor.config import Settings
from world_cup_predictor.simulation.monte_carlo import Matchup, simulate_knockout_tournament


st.set_page_config(page_title="World Cup Predictor", layout="wide")
st.title("World Cup Predictor")

settings = Settings()
st.write(f"Simulations: {settings.n_simulations}")

home_team = st.text_input("Home team", "Team A")
away_team = st.text_input("Away team", "Team B")
home_win_probability = st.slider("Home win probability", 0.0, 1.0, 0.5)

if st.button("Simulate"):
    probs = simulate_knockout_tournament(
        [Matchup(home_team=home_team, away_team=away_team, home_win_probability=home_win_probability)],
        n_simulations=settings.n_simulations,
    )
    st.write(probs)
