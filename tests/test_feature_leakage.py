"""The single most important test file in this project.

Every test here follows the same pattern: compute a feature on the full
dataset, then again on a dataset truncated to end right after some earlier
match, and assert the earlier match's feature value is *identical* either
way. If a feature secretly depends on the future, truncating the future
changes its value -- that's the whole test.
"""

from __future__ import annotations

import pandas as pd

from worldcup_predictor.features.elo import compute_elo_ratings, initial_ratings_from_teams
from worldcup_predictor.features.group_stage import attach_group_stage_features, compute_group_stage_summary
from worldcup_predictor.features.rolling import compute_rolling_form


def _truncate_before_final_round(matches: pd.DataFrame) -> pd.DataFrame:
    """Drop the two still-open matches (third place + final, match_id 15/16)."""
    return matches[matches["match_id"].astype(int) <= 14].sort_values("date").reset_index(drop=True)


def test_elo_pre_match_rating_ignores_future_matches(matches, raw_wc2026, features_config):
    initial = initial_ratings_from_teams(raw_wc2026["teams"])
    full = compute_elo_ratings(matches, initial, features_config.elo)
    truncated = compute_elo_ratings(_truncate_before_final_round(matches), initial, features_config.elo)

    merged = truncated.merge(full, on="match_id", suffixes=("_truncated", "_full"))
    assert len(merged) == len(truncated)
    pd.testing.assert_series_equal(
        merged["home_elo_pre_truncated"], merged["home_elo_pre_full"], check_names=False
    )
    pd.testing.assert_series_equal(
        merged["away_elo_pre_truncated"], merged["away_elo_pre_full"], check_names=False
    )


def test_rolling_form_ignores_future_matches(matches, features_config):
    windows = features_config.rolling_form.windows
    full = compute_rolling_form(matches, windows)
    truncated = compute_rolling_form(_truncate_before_final_round(matches), windows)

    merged = truncated.merge(full, on="match_id", suffixes=("_truncated", "_full"))
    col = f"home_form{windows[0]}_win_rate"
    pd.testing.assert_series_equal(
        merged[f"{col}_truncated"], merged[f"{col}_full"], check_names=False
    )


def test_group_stage_features_are_attached_only_to_knockout_rows(feature_frame):
    group_rows = feature_frame[feature_frame["stage"] == "group_stage"]
    knockout_rows = feature_frame[feature_frame["is_knockout"]]

    assert group_rows["home_group_points"].isna().all()
    assert group_rows["away_group_points"].isna().all()
    assert knockout_rows["home_group_points"].notna().all()
    assert knockout_rows["away_group_points"].notna().all()


def test_group_stage_summary_only_uses_completed_group_matches(matches, raw_wc2026):
    teams = raw_wc2026["teams"]
    team_groups = dict(zip(teams["team_name"], teams["group_letter"], strict=True))
    summary = compute_group_stage_summary(matches, team_groups)
    # 12 completed group matches per group of 4 (6 matches x 2 groups), all teams present exactly once
    assert summary["team"].nunique() == len(summary)
    assert set(summary["team"]) == set(matches.loc[matches["stage"] == "group_stage", "home_team"]) | set(
        matches.loc[matches["stage"] == "group_stage", "away_team"]
    )
    # Attaching must not mutate the summary computed from a truncated frame
    truncated_summary = compute_group_stage_summary(_truncate_before_final_round(matches), team_groups)
    pd.testing.assert_frame_equal(summary, truncated_summary)


def test_attach_group_stage_features_does_not_leak_into_group_stage_rows(matches, raw_wc2026):
    teams = raw_wc2026["teams"]
    team_groups = dict(zip(teams["team_name"], teams["group_letter"], strict=True))
    summary = compute_group_stage_summary(matches, team_groups)
    attached = attach_group_stage_features(matches, summary)
    assert attached.loc[attached["stage"] == "group_stage", "home_group_rank"].isna().all()


def test_model_configs_never_use_the_matchs_own_xg_as_a_feature(model_config):
    """Leakage rule #4: this match's own xG must never predict this match's own result."""
    forbidden = {"home_xg", "away_xg", "home_score", "away_score", "winner"}
    assert forbidden.isdisjoint(model_config.goal_model.features)
    assert forbidden.isdisjoint(model_config.outcome_model.features)


def test_rest_days_ignore_future_matches(matches, features_config):
    from worldcup_predictor.features.feature_pipeline import compute_rest_days

    full = compute_rest_days(matches, features_config.rest_days.cap_days)
    truncated = compute_rest_days(_truncate_before_final_round(matches), features_config.rest_days.cap_days)
    merged = truncated.merge(full, on="match_id", suffixes=("_truncated", "_full"))
    pd.testing.assert_series_equal(
        merged["home_rest_days_truncated"], merged["home_rest_days_full"], check_names=False
    )
