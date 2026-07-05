"""Plotly figure builders shared by the Streamlit dashboard and (optionally)
static report exports. Kept dependency-free of Streamlit itself so these are
independently testable."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def champion_probability_bar(probability_table: pd.DataFrame, top_n: int = 15) -> go.Figure:
    data = probability_table.sort_values("prob_champion", ascending=False).head(top_n)
    fig = px.bar(
        data,
        x="prob_champion",
        y="team",
        orientation="h",
        title="Championship probability",
        labels={"prob_champion": "Probability", "team": ""},
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, xaxis_tickformat=".0%")
    return fig


def stage_progression_chart(probability_table: pd.DataFrame, stages: list[str], top_n: int = 8) -> go.Figure:
    ranked = probability_table.sort_values("prob_champion", ascending=False).head(top_n)
    columns = [f"prob_{stage}" for stage in stages if f"prob_{stage}" in ranked.columns]
    long = ranked.melt(id_vars="team", value_vars=columns, var_name="stage", value_name="probability")
    long["stage"] = long["stage"].str.removeprefix("prob_")
    fig = px.bar(
        long,
        x="team",
        y="probability",
        color="stage",
        barmode="group",
        title="Stage-reach probability by team",
    )
    fig.update_layout(yaxis_tickformat=".0%")
    return fig


def match_outcome_comparison(prediction_row: pd.Series) -> go.Figure:
    models = {
        "Outcome model": ("outcome_model_p_home", "outcome_model_p_draw", "outcome_model_p_away"),
        "Goal model (Poisson)": ("goal_model_p_home", "goal_model_p_draw", "goal_model_p_away"),
        "FIFA ranking baseline": ("fifa_ranking_p_home", "fifa_ranking_p_draw", "fifa_ranking_p_away"),
        "Elo baseline": ("elo_diff_p_home", "elo_diff_p_draw", "elo_diff_p_away"),
    }
    fig = go.Figure()
    for label, (h, d, a) in models.items():
        values = [prediction_row[h], prediction_row[d], prediction_row[a]]
        fig.add_trace(go.Bar(name=label, x=["Home win", "Draw", "Away win"], y=values))
    fig.update_layout(barmode="group", title="Model comparison", yaxis_tickformat=".0%")
    return fig
