"""Orchestrates raw -> interim -> model-ready feature frame.

This is the single place that wires the individually-testable feature
modules (elo, rolling, group_stage, squad) together in the correct order.
Anything added here must preserve the leakage invariant: a feature value
attached to match M may only be derived from data available strictly before
M's kickoff.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from worldcup_predictor.data.preprocessing import build_match_table
from worldcup_predictor.features.elo import compute_elo_ratings, initial_ratings_from_teams
from worldcup_predictor.features.group_stage import (
    attach_group_stage_features,
    compute_group_stage_summary,
)
from worldcup_predictor.features.rolling import compute_rolling_form
from worldcup_predictor.features.squad import attach_squad_features, compute_squad_features
from worldcup_predictor.utils.config import FeaturesConfig


def compute_rest_days(matches: pd.DataFrame, cap_days: int) -> pd.DataFrame:
    """Days since each team's last completed match, capped to avoid outliers.

    A team's actual rest before its very first match in the dataset is
    unknown (we don't have its pre-tournament friendly/qualifier schedule),
    so it is explicitly set to ``cap_days`` (the "fully rested" ceiling)
    rather than left as NaN -- a documented assumption, not a silent one
    (see README limitations), which also keeps every played match usable for
    model training instead of being dropped for a missing feature.
    """
    last_played: dict[str, dt.date] = {}
    rows: list[dict] = []

    for row in matches.itertuples(index=False):
        match_date = row.date.date()
        row_data: dict = {"match_id": row.match_id}
        for side, team in (("home", row.home_team), ("away", row.away_team)):
            previous = last_played.get(team)
            row_data[f"{side}_rest_days"] = (
                min((match_date - previous).days, cap_days) if previous is not None else cap_days
            )
        rows.append(row_data)

        if pd.notna(row.winner):
            last_played[row.home_team] = match_date
            last_played[row.away_team] = match_date

    return pd.DataFrame(rows)


def _add_form_diff_columns(df: pd.DataFrame, windows: list[int], metrics: list[str]) -> pd.DataFrame:
    for window in windows:
        for metric in metrics:
            col = f"form{window}_{metric}_diff"
            home_col, away_col = f"home_form{window}_{metric}", f"away_form{window}_{metric}"
            df[col] = df[home_col].astype(float) - df[away_col].astype(float)
    return df


def build_feature_frame(
    raw: dict[str, pd.DataFrame],
    config: FeaturesConfig,
    tournament: str = "FIFA World Cup 2026",
) -> pd.DataFrame:
    """Build the full, model-ready, leakage-safe feature frame for every match."""
    matches = build_match_table(raw, tournament=tournament)
    teams = raw["teams"]

    elo_df = compute_elo_ratings(matches, initial_ratings_from_teams(teams), config.elo)
    form_df = compute_rolling_form(matches, config.rolling_form.windows)
    rest_df = compute_rest_days(matches, config.rest_days.cap_days)

    result = matches.merge(elo_df, on="match_id").merge(form_df, on="match_id").merge(rest_df, on="match_id")
    result = _add_form_diff_columns(result, config.rolling_form.windows, config.rolling_form.metrics)

    if config.group_stage_features.enabled:
        team_groups = dict(zip(teams["team_name"], teams["group_letter"], strict=True))
        group_summary = compute_group_stage_summary(matches, team_groups)
        if not group_summary.empty:
            result = attach_group_stage_features(result, group_summary)

    reference_date = matches["date"].min().date()
    squad_summary = compute_squad_features(
        raw["squads_and_players"], teams, reference_date, config.squad_features
    )
    result = attach_squad_features(result, squad_summary)

    result["ranking_diff"] = result["away_fifa_ranking"].astype(float) - result[
        "home_fifa_ranking"
    ].astype(float)  # positive => home team has the better (lower) ranking
    result["rest_day_diff"] = result["home_rest_days"].astype(float) - result["away_rest_days"].astype(
        float
    )
    result["is_neutral"] = result["neutral_venue"].astype(int)
    if not config.venue_features.use_altitude:
        result["venue_elevation_m"] = np.nan

    return result


def build_historical_feature_frame(historical_matches: pd.DataFrame, config: FeaturesConfig) -> pd.DataFrame:
    """Feature frame for the historical (1930-2022) dataset: Elo + rolling
    form + rest days only.

    Unlike ``build_feature_frame``, there is no squad market value, no
    per-match FIFA ranking, and no group-stage-qualification context this far
    back -- those data simply don't exist for most of 1930-2022 (see README
    limitations). This is why the historical backtest in
    ``models.evaluation.run_time_based_backtest`` uses a reduced feature list
    (``configs/model.yaml: historical_backtest``) rather than the full 2026
    feature set.

    Elo starts from the configured ``initial_rating`` for every team's first
    appearance in 1930 -- there is no external pre-1930 rating to warm-start
    from, unlike the wc2026 dataset's ``teams.csv`` (which ships an
    externally-sourced current rating). This Elo chain is intentionally kept
    separate from the wc2026 Elo chain (different scale/coverage: WC-only
    matches across 96 years vs. an externally-calibrated current rating), so
    it must only be used for backtesting on this historical dataset itself,
    never spliced into 2026 predictions.
    """
    elo_df = compute_elo_ratings(historical_matches, {}, config.elo)
    form_df = compute_rolling_form(historical_matches, config.rolling_form.windows)
    rest_df = compute_rest_days(historical_matches, config.rest_days.cap_days)

    result = (
        historical_matches.merge(elo_df, on="match_id")
        .merge(form_df, on="match_id")
        .merge(rest_df, on="match_id")
    )
    result = _add_form_diff_columns(result, config.rolling_form.windows, config.rolling_form.metrics)
    result["rest_day_diff"] = result["home_rest_days"].astype(float) - result["away_rest_days"].astype(
        float
    )
    result["is_neutral"] = result["neutral_venue"].astype(int)
    return result
