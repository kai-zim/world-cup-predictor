"""Chronological Elo rating engine.

The rating used to predict a match is always the rating computed from every
*earlier* match only -- this module produces both the pre-match rating (the
feature) and the post-match rating (next match's pre-match rating) in a
single left-to-right pass, which is what makes it leakage-safe by
construction rather than by convention.
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from worldcup_predictor.data.validation import assert_chronological
from worldcup_predictor.utils.config import EloConfig


def _expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** (-(rating_a - rating_b) / 400.0))


def compute_elo_ratings(
    matches: pd.DataFrame,
    initial_ratings: Mapping[str, float],
    config: EloConfig,
) -> pd.DataFrame:
    """Compute pre- and post-match Elo ratings for every match, in order.

    ``matches`` must be sorted chronologically and contain at least:
    match_id, date, home_team, away_team, is_knockout, neutral_venue,
    home_score, away_score, winner (None for matches not yet played).

    Unplayed matches get pre-match ratings (the current state of the team)
    but do not update anyone's rating.
    """
    assert_chronological(matches, "date")

    ratings: dict[str, float] = dict(initial_ratings)
    rows: list[dict] = []

    for row in matches.itertuples(index=False):
        home, away = row.home_team, row.away_team
        rating_home = ratings.get(home, config.initial_rating)
        rating_away = ratings.get(away, config.initial_rating)

        played = pd.notna(row.winner)
        home_advantage = 0.0 if row.neutral_venue else config.home_advantage
        expected_home = _expected_score(rating_home + home_advantage, rating_away)

        new_home, new_away = rating_home, rating_away
        if played:
            k_factor = config.k_factor_knockout if row.is_knockout else config.k_factor_group_stage
            if row.winner == "home":
                actual_home = 1.0
            elif row.winner == "away":
                actual_home = 0.0
            else:
                actual_home = 0.5
            new_home = rating_home + k_factor * (actual_home - expected_home)
            new_away = rating_away + k_factor * ((1 - actual_home) - (1 - expected_home))
            ratings[home] = new_home
            ratings[away] = new_away

        rows.append(
            {
                "match_id": row.match_id,
                "home_elo_pre": rating_home,
                "away_elo_pre": rating_away,
                "home_elo_post": new_home,
                "away_elo_post": new_away,
                "elo_diff": rating_home - rating_away,
                "elo_expected_home_win_prob": expected_home,
            }
        )

    return pd.DataFrame(rows)


def initial_ratings_from_teams(teams: pd.DataFrame) -> dict[str, float]:
    """Build the {team_name: elo_rating} starting map from teams.csv.

    This is the pre-tournament rating the data provider ships; it becomes the
    seed for the chronological update above, not a static feature by itself.
    """
    return dict(zip(teams["team_name"], teams["elo_rating"].astype(float), strict=True))
