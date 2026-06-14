"""Per-match prediction utilities.

Exposes lower-level fixture prediction functions used by the CLI, dashboard,
and reporting layer. The full tournament forecast is in pipeline.run_forecast;
these helpers cover individual match W/D/L and expected-goals prediction.
"""

from __future__ import annotations

import pandas as pd

from wc2026.models.goal_model import GoalModel, outcome_probabilities, scoreline_matrix


def predict_match(
    home_team: str,
    away_team: str,
    home_elo: float,
    away_elo: float,
    model: GoalModel,
    neutral: bool = True,
    max_goals: int = 10,
) -> dict[str, object]:
    """Predict W/D/L probabilities and expected goals for a single fixture.

    Works with any GoalModel backend (analytic or learned). Knockout fixtures
    should always be called with neutral=True since they are played at a fixed
    third-country venue.

    Args:
        home_team: Name of the nominally 'home' team (bracket left side).
        away_team: Name of the nominally 'away' team (bracket right side).
        home_elo: Current Elo rating of home_team.
        away_elo: Current Elo rating of away_team.
        model: GoalModel instance (EloPoissonModel or LearnedGoalModel).
        neutral: Whether the match is on a neutral venue.
        max_goals: Scoreline grid truncation (higher = more precise but slower).

    Returns:
        Dict with keys: home_team, away_team, p_home_win, p_draw, p_away_win,
        lambda_home, lambda_away.
    """
    lam_h, lam_a = model.expected_goals(home_elo, away_elo, neutral=neutral)
    mat = scoreline_matrix(lam_h, lam_a, max_goals=max_goals)
    p_home, p_draw, p_away = outcome_probabilities(mat)
    return {
        "home_team": home_team,
        "away_team": away_team,
        "p_home_win": round(p_home, 4),
        "p_draw": round(p_draw, 4),
        "p_away_win": round(p_away, 4),
        "lambda_home": round(lam_h, 3),
        "lambda_away": round(lam_a, 3),
    }


def predict_batch(
    fixtures: pd.DataFrame,
    model: GoalModel,
    elo_ratings: dict[str, float],
    neutral: bool = True,
    max_goals: int = 10,
) -> pd.DataFrame:
    """Predict W/D/L for a batch of fixtures.

    Args:
        fixtures: DataFrame with columns home_team and away_team. If
            home_elo_pre / away_elo_pre columns are present they take
            precedence over the elo_ratings lookup.
        model: GoalModel backend.
        elo_ratings: Fallback Elo lookup when pre-computed columns are absent.
        neutral: Applied to all fixtures; set False for qualifying/friendlies.
        max_goals: Scoreline grid truncation.

    Returns:
        DataFrame with one row per fixture and W/D/L + xG columns.
    """
    records = []
    for row in fixtures.itertuples(index=False):
        h_elo = float(getattr(row, "home_elo_pre", None) or elo_ratings.get(row.home_team, 1500.0))
        a_elo = float(getattr(row, "away_elo_pre", None) or elo_ratings.get(row.away_team, 1500.0))
        records.append(
            predict_match(
                home_team=row.home_team,
                away_team=row.away_team,
                home_elo=h_elo,
                away_elo=a_elo,
                model=model,
                neutral=neutral,
                max_goals=max_goals,
            )
        )
    return pd.DataFrame(records)