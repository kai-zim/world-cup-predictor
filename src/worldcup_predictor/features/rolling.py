"""Rolling-form features over each team's last N matches.

For every match, a team's form features are computed from strictly earlier
matches only (the team's history is updated *after* the feature row is
emitted), and windows shorter than the requested size use whatever history
exists rather than being padded with fabricated data -- a team's first
tournament match has no form history at all (``None``), by design.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import pandas as pd

from worldcup_predictor.data.validation import assert_chronological


def _team_match_result(goals_for: float, goals_against: float, xg_for, xg_against) -> dict[str, Any]:
    if goals_for > goals_against:
        points = 3
    elif goals_for == goals_against:
        points = 1
    else:
        points = 0
    return {
        "points": points,
        "goals_for": goals_for,
        "goals_against": goals_against,
        "xg_for": xg_for,
        "xg_against": xg_against,
        "clean_sheet": goals_against == 0,
    }


def compute_rolling_form(matches: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """Return one row per match_id with home_/away_ form{window}_{metric} columns."""
    assert_chronological(matches, "date")

    history: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rows: list[dict[str, Any]] = []

    for row in matches.itertuples(index=False):
        result_row: dict[str, Any] = {"match_id": row.match_id}

        for side, team in (("home", row.home_team), ("away", row.away_team)):
            team_history = history[team]
            for window in windows:
                recent = team_history[-window:]
                n = len(recent)
                prefix = f"{side}_form{window}_"
                if n == 0:
                    for metric in (
                        "win_rate",
                        "points_per_game",
                        "goals_per_game",
                        "goals_against_per_game",
                        "goal_diff_per_game",
                        "xg_per_game",
                        "xga_per_game",
                        "clean_sheet_rate",
                        "loss_rate",
                    ):
                        result_row[prefix + metric] = None
                    continue

                wins = sum(1 for r in recent if r["points"] == 3)
                losses = sum(1 for r in recent if r["points"] == 0)
                goals_for = sum(r["goals_for"] for r in recent)
                goals_against = sum(r["goals_against"] for r in recent)
                clean_sheets = sum(1 for r in recent if r["clean_sheet"])
                xg_values = [r["xg_for"] for r in recent if pd.notna(r["xg_for"])]
                xga_values = [r["xg_against"] for r in recent if pd.notna(r["xg_against"])]

                result_row[prefix + "win_rate"] = wins / n
                result_row[prefix + "points_per_game"] = sum(r["points"] for r in recent) / n
                result_row[prefix + "goals_per_game"] = goals_for / n
                result_row[prefix + "goals_against_per_game"] = goals_against / n
                result_row[prefix + "goal_diff_per_game"] = (goals_for - goals_against) / n
                result_row[prefix + "xg_per_game"] = (sum(xg_values) / len(xg_values)) if xg_values else None
                result_row[prefix + "xga_per_game"] = (
                    (sum(xga_values) / len(xga_values)) if xga_values else None
                )
                result_row[prefix + "clean_sheet_rate"] = clean_sheets / n
                result_row[prefix + "loss_rate"] = losses / n

        rows.append(result_row)

        if pd.notna(row.winner):
            home_result = _team_match_result(row.home_score, row.away_score, row.home_xg, row.away_xg)
            away_result = _team_match_result(row.away_score, row.home_score, row.away_xg, row.home_xg)
            history[row.home_team].append(home_result)
            history[row.away_team].append(away_result)

    return pd.DataFrame(rows)
