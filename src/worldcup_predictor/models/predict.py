"""Turn fitted models into match-outcome predictions.

Kept separate from train.py so the CLI can retrain once and predict many
times (e.g. all still-open knockout matches) without re-fitting.
"""

from __future__ import annotations

import pandas as pd

from worldcup_predictor.models.baselines import (
    EloDiffBaseline,
    FifaRankingBaseline,
    HistoricalWinRateBaseline,
    PoissonBaseline,
    poisson_match_outcome_probs,
)
from worldcup_predictor.models.train import GoalModel, OutcomeModel


def predict_goal_model(model: GoalModel, df: pd.DataFrame, max_goals: int = 10) -> pd.DataFrame:
    """Expected goals + the 1x2 probabilities implied by them."""
    lambda_home, lambda_away = model.predict(df)
    proba = poisson_match_outcome_probs(lambda_home, lambda_away, max_goals)
    proba.index = df.index
    proba["expected_home_goals"] = lambda_home
    proba["expected_away_goals"] = lambda_away
    return proba


def predict_outcome_model(model: OutcomeModel, df: pd.DataFrame) -> pd.DataFrame:
    return model.predict_proba(df)


def build_predictions_table(
    df: pd.DataFrame,
    goal_model: GoalModel,
    outcome_model: OutcomeModel,
    historical_baseline: HistoricalWinRateBaseline,
) -> pd.DataFrame:
    """One row per match with every model's 1x2 probabilities side by side,
    for CLI output / reporting / the dashboard's match predictor page."""
    match_columns = [
        "match_id", "date", "stage", "is_knockout",
        "home_team", "away_team", "home_score", "away_score", "winner",
    ]
    base_cols = df[match_columns].reset_index(drop=True)

    goal_pred = predict_goal_model(goal_model, df).reset_index(drop=True).add_prefix("goal_model_")
    outcome_pred = (
        predict_outcome_model(outcome_model, df).reset_index(drop=True).add_prefix("outcome_model_")
    )
    fifa_pred = FifaRankingBaseline().predict_proba(df).reset_index(drop=True).add_prefix("fifa_ranking_")
    elo_pred = EloDiffBaseline().predict_proba(df).reset_index(drop=True).add_prefix("elo_diff_")
    poisson_pred = PoissonBaseline().predict_proba(df).reset_index(drop=True).add_prefix("poisson_")
    hist_pred = historical_baseline.predict_proba(df).reset_index(drop=True).add_prefix("historical_")

    return pd.concat(
        [base_cols, outcome_pred, goal_pred, fifa_pred, elo_pred, poisson_pred, hist_pred], axis=1
    )
