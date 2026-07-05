"""Squad-level features from squads_and_players.csv.

Squad composition is treated as constant for the whole tournament (the MVP
does not use match_lineups.csv for a per-match starting XI -- see README
roadmap), so these features are computed once per team and joined onto every
match that team plays.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from worldcup_predictor.utils.config import SquadFeaturesConfig


def _position_balance_entropy(position_counts: pd.Series) -> float | None:
    """Shannon entropy of the squad's position distribution (bits).

    Higher = more evenly spread across GK/DEF/MID/FWD; 0 = every player has
    the same listed position (degenerate data, not a realistic squad).
    """
    if position_counts.empty:
        return None
    probabilities = position_counts.to_numpy(dtype=float)
    return float(-(probabilities * np.log2(probabilities)).sum())


def _age_years(date_of_birth: str | None, reference_date: dt.date) -> float | None:
    if not date_of_birth or pd.isna(date_of_birth):
        return None
    dob = pd.to_datetime(date_of_birth).date()
    return (reference_date - dob).days / 365.25


def compute_squad_features(
    squads: pd.DataFrame,
    teams: pd.DataFrame,
    reference_date: dt.date,
    config: SquadFeaturesConfig,
) -> pd.DataFrame:
    """One row per team: market value, age, caps and position-balance features."""
    merged = squads.merge(teams[["team_id", "team_name"]], on="team_id", how="left")
    merged["age_years"] = merged["date_of_birth"].apply(lambda d: _age_years(d, reference_date))

    records = []
    for team, group in merged.groupby("team_name"):
        market_values = group["market_value_eur"].dropna()
        top_n = market_values.sort_values(ascending=False).head(config.top_n_market_value)
        caps = group["caps"].dropna()
        position_counts = group["position"].value_counts(normalize=True)

        records.append(
            {
                "team": team,
                "squad_size": len(group),
                "squad_total_market_value_eur": market_values.sum(),
                "squad_avg_market_value_eur": market_values.mean() if len(market_values) else None,
                "squad_top11_market_value_eur": top_n.sum() if len(top_n) else None,
                "squad_avg_age": group["age_years"].mean(skipna=True),
                "squad_avg_caps": caps.mean() if len(caps) else None,
                "squad_players_over_caps_threshold": int((caps > config.caps_threshold).sum()),
                "squad_position_balance_entropy": _position_balance_entropy(position_counts),
            }
        )

    return pd.DataFrame(records)


_ATTACH_COLUMNS = [
    "squad_size",
    "squad_total_market_value_eur",
    "squad_avg_market_value_eur",
    "squad_top11_market_value_eur",
    "squad_avg_age",
    "squad_avg_caps",
    "squad_players_over_caps_threshold",
    "squad_position_balance_entropy",
]


def attach_squad_features(matches: pd.DataFrame, squad_summary: pd.DataFrame) -> pd.DataFrame:
    result = matches.copy()
    for prefix, team_col in (("home", "home_team"), ("away", "away_team")):
        renamed = squad_summary[["team", *_ATTACH_COLUMNS]].rename(
            columns={col: f"{prefix}_{col}" for col in _ATTACH_COLUMNS}
        )
        result = result.merge(renamed, left_on=team_col, right_on="team", how="left").drop(columns=["team"])
    result["market_value_diff_log"] = _log_ratio(
        result["home_squad_total_market_value_eur"], result["away_squad_total_market_value_eur"]
    )
    return result


def _log_ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    import numpy as np

    return np.log((a.astype(float) + 1.0) / (b.astype(float) + 1.0))
