"""Group-stage summary features.

Rule enforced here, not just documented: these features are computed once
from the completed group stage and then attached *only* to knockout-stage
rows. Group-stage matches themselves never see their own group's summary,
since that summary is not fully known until the group stage ends.
"""

from __future__ import annotations

import pandas as pd


def compute_group_stage_summary(
    matches: pd.DataFrame, team_groups: dict[str, str], n_best_third_qualify: int = 8
) -> pd.DataFrame:
    """Per-team group-stage table: points, GD, rank, qualification.

    ``team_groups`` maps team_name -> group_letter (from teams.csv). Ranking
    tiebreak is points -> goal difference -> goals for (FIFA also uses
    head-to-head results and disciplinary points first; that finer tiebreak
    is not modeled here, see README limitations).
    """
    group_matches = matches[matches["stage"] == "group_stage"]
    completed = group_matches[group_matches["home_score"].notna() & group_matches["away_score"].notna()]

    teams = sorted(set(completed["home_team"]) | set(completed["away_team"]))
    records = []
    for team in teams:
        home_rows = completed[completed["home_team"] == team]
        away_rows = completed[completed["away_team"] == team]

        wins = int((home_rows["home_score"] > home_rows["away_score"]).sum()) + int(
            (away_rows["away_score"] > away_rows["home_score"]).sum()
        )
        draws = int((home_rows["home_score"] == home_rows["away_score"]).sum()) + int(
            (away_rows["away_score"] == away_rows["home_score"]).sum()
        )
        played = len(home_rows) + len(away_rows)
        losses = played - wins - draws
        goals_for = home_rows["home_score"].sum() + away_rows["away_score"].sum()
        goals_against = home_rows["away_score"].sum() + away_rows["home_score"].sum()
        xg_for = home_rows["home_xg"].sum(skipna=True) + away_rows["away_xg"].sum(skipna=True)
        xg_against = home_rows["away_xg"].sum(skipna=True) + away_rows["home_xg"].sum(skipna=True)

        records.append(
            {
                "team": team,
                "group": team_groups.get(team),
                "group_played": played,
                "group_wins": wins,
                "group_draws": draws,
                "group_losses": losses,
                "group_points": wins * 3 + draws,
                "group_goals_for": goals_for,
                "group_goals_against": goals_against,
                "group_goal_diff": goals_for - goals_against,
                "group_xg_for": xg_for,
                "group_xg_against": xg_against,
            }
        )

    summary = pd.DataFrame(records)
    if summary.empty:
        return summary

    summary = summary.sort_values(
        ["group", "group_points", "group_goal_diff", "group_goals_for"],
        ascending=[True, False, False, False],
    )
    summary["group_rank"] = summary.groupby("group").cumcount() + 1
    summary["qualified_group_winner"] = summary["group_rank"] == 1
    summary["qualified_runner_up"] = summary["group_rank"] == 2

    thirds = summary[summary["group_rank"] == 3].sort_values(
        ["group_points", "group_goal_diff", "group_goals_for"], ascending=False
    )
    best_third_teams = set(thirds["team"].head(n_best_third_qualify))
    summary["qualified_best_third"] = summary["team"].isin(best_third_teams) & (
        summary["group_rank"] == 3
    )
    summary["qualified_for_knockout"] = (
        summary["qualified_group_winner"] | summary["qualified_runner_up"] | summary["qualified_best_third"]
    )
    return summary.reset_index(drop=True)


_ATTACH_COLUMNS = [
    "group_points",
    "group_wins",
    "group_draws",
    "group_losses",
    "group_goals_for",
    "group_goals_against",
    "group_goal_diff",
    "group_xg_for",
    "group_xg_against",
    "group_rank",
    "qualified_group_winner",
    "qualified_runner_up",
    "qualified_best_third",
]


def attach_group_stage_features(matches: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """Left-join group summary onto knockout rows only (home_/away_ prefixed)."""
    result = matches.copy()
    for prefix, team_col in (("home", "home_team"), ("away", "away_team")):
        renamed = summary[["team", *_ATTACH_COLUMNS]].rename(
            columns={col: f"{prefix}_{col}" for col in _ATTACH_COLUMNS}
        )
        result = result.merge(renamed, left_on=team_col, right_on="team", how="left").drop(columns=["team"])

    is_group_stage = result["stage"] == "group_stage"
    for prefix in ("home", "away"):
        for col in _ATTACH_COLUMNS:
            full_col = f"{prefix}_{col}"
            result[full_col] = result[full_col].astype(object)
            result.loc[is_group_stage, full_col] = None
    return result
