"""Tests for the historical (piterfm, 1930-2022) dataset integration:
normalization into the canonical schema, the reduced Elo/form-only feature
frame, and the time-based backtest itself.
"""

from __future__ import annotations

import pandas as pd
import pytest

from worldcup_predictor.data.preprocessing import build_historical_match_table
from worldcup_predictor.models.evaluation import run_time_based_backtest


def test_historical_match_table_has_canonical_columns(historical_matches):
    for col in ["match_id", "date", "season", "stage", "is_knockout", "home_team", "away_team", "winner"]:
        assert col in historical_matches.columns
    assert len(historical_matches) == 12


def test_historical_matches_are_all_played(historical_matches):
    # Unlike the wc2026 dataset, every historical match is a completed result.
    assert historical_matches["winner"].notna().all()
    assert historical_matches["home_score"].notna().all()


def test_historical_stage_mapping_and_knockout_flag(historical_matches):
    group_rows = historical_matches[historical_matches["stage"] == "group_stage"]
    final_rows = historical_matches[historical_matches["stage"] == "final"]
    assert len(final_rows) == 6  # one per World Cup year in the fixture
    assert not group_rows["is_knockout"].any()
    assert final_rows["is_knockout"].all()


def test_penalty_shootout_detected_in_historical_data(historical_matches):
    row = historical_matches[
        (historical_matches["season"] == 2006) & (historical_matches["stage"] == "final")
    ].iloc[0]
    assert row["went_to_penalties"]
    assert row["went_to_extra_time"]
    assert row["home_penalty_score"] > row["away_penalty_score"]
    assert row["winner"] == "home"  # Italy (home) won on penalties


def test_extra_time_detected_without_penalties(historical_matches):
    row = historical_matches[
        (historical_matches["season"] == 2014) & (historical_matches["stage"] == "final")
    ].iloc[0]
    assert row["went_to_extra_time"]
    assert not row["went_to_penalties"]
    assert row["winner"] == "home"  # Germany won 1-0 in extra time


def test_host_nation_match_is_not_neutral(historical_matches):
    # 1998 final: Brazil vs France, hosted by France -> France (away) is at home.
    row = historical_matches[
        (historical_matches["season"] == 1998) & (historical_matches["stage"] == "final")
    ].iloc[0]
    assert not row["neutral_venue"]

    # 2002 final: Brazil vs Germany, co-hosted by South Korea/Japan -> neutral for both.
    row = historical_matches[
        (historical_matches["season"] == 2002) & (historical_matches["stage"] == "final")
    ].iloc[0]
    assert row["neutral_venue"]


def test_unrecognized_round_value_raises(raw_historical):
    tampered = raw_historical.copy()
    tampered.loc[0, "Round"] = "Some Unknown Round"
    with pytest.raises(ValueError, match="Unrecognized Round value"):
        build_historical_match_table(tampered)


def test_historical_feature_frame_has_elo_and_form_columns(historical_feature_frame):
    for col in ["home_elo_pre", "away_elo_pre", "elo_diff", "home_form5_win_rate", "rest_day_diff"]:
        assert col in historical_feature_frame.columns
    # No squad/ranking features -- not available for 1930-2022.
    assert "home_squad_total_market_value_eur" not in historical_feature_frame.columns
    assert "home_fifa_ranking" in historical_feature_frame.columns
    assert historical_feature_frame["home_fifa_ranking"].isna().all()


def test_historical_elo_ignores_future_matches(historical_matches, features_config):
    from worldcup_predictor.features.elo import compute_elo_ratings

    full = compute_elo_ratings(historical_matches, {}, features_config.elo)
    truncated_matches = historical_matches[historical_matches["season"] <= 2014].reset_index(drop=True)
    truncated = compute_elo_ratings(truncated_matches, {}, features_config.elo)

    merged = truncated.merge(full, on="match_id", suffixes=("_truncated", "_full"))
    pd.testing.assert_series_equal(
        merged["home_elo_pre_truncated"], merged["home_elo_pre_full"], check_names=False
    )


def test_run_time_based_backtest_produces_metrics_per_split_and_model(historical_feature_frame, model_config):
    results = run_time_based_backtest(historical_feature_frame, model_config)
    assert not results.empty
    assert {"split", "model", "accuracy", "log_loss", "brier_score", "rps"}.issubset(results.columns)
    # Every configured model/baseline should appear for every split that produced results.
    expected_models = {
        "elo_diff_baseline",
        "poisson_baseline",
        "historical_win_rate_baseline",
        "goal_model",
        "outcome_model",
    }
    for _, group in results.groupby("split"):
        assert set(group["model"]) == expected_models
    assert (results["accuracy"] >= 0).all() and (results["accuracy"] <= 1).all()


def test_run_time_based_backtest_skips_splits_with_no_data(historical_feature_frame, model_config):
    results = run_time_based_backtest(historical_feature_frame, model_config)
    # The 2026 split (train_until=2022, test_year=2026) has no historical rows for
    # 2026 and must be silently-but-visibly skipped, not produce empty/garbage metrics.
    assert "train<=2022->test=2026" not in set(results["split"])
