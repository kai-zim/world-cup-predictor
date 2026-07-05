from pathlib import Path

import pandas as pd
import pytest

from worldcup_predictor.data.loaders import load_raw_historical
from worldcup_predictor.data.schemas import MatchRecord
from worldcup_predictor.data.validation import (
    assert_chronological,
    assert_no_future_leakage,
    missing_value_report,
)

PITERFM_FIXTURE = Path(__file__).parent / "fixtures" / "piterfm_sample.csv"


def test_raw_fixtures_validate_against_schemas(raw_wc2026):
    # If loading (in the conftest fixture) had failed schema validation, this
    # session-scoped fixture would already have raised -- this test just
    # asserts the expected tables came through.
    for name in ["teams", "venues", "tournament_stages", "matches", "squads_and_players", "match_team_stats"]:
        assert name in raw_wc2026
        assert len(raw_wc2026[name]) > 0


def test_build_match_table_has_canonical_columns(matches):
    for col in ["match_id", "date", "stage", "home_team", "away_team", "winner", "is_knockout"]:
        assert col in matches.columns
    assert len(matches) == 16


def test_group_stage_matches_are_never_marked_knockout(matches):
    assert not matches.loc[matches["stage"] == "group_stage", "is_knockout"].any()


def test_wc2026_knockout_matches_never_end_in_a_draw(matches):
    """Business rule of the *current* format specifically (always penalties on
    a draw) -- not a universal rule, see test_matchrecord_allows_pre_shootout_era_knockout_draws."""
    knockout = matches[matches["is_knockout"] & matches["winner"].notna()]
    assert (knockout["winner"] != "draw").all()


def test_penalty_shootout_is_recorded_and_resolved(matches):
    row = matches.loc[matches["match_id"] == "14"].iloc[0]
    assert row["went_to_penalties"]
    assert row["home_penalty_score"] > row["away_penalty_score"]
    assert row["winner"] == "home"


def test_unplayed_matches_have_no_winner(matches):
    unplayed = matches[matches["home_score"].isna()]
    assert unplayed["winner"].isna().all()
    assert set(unplayed["match_id"]) == {"15", "16"}


def test_matchrecord_allows_pre_shootout_era_knockout_draws():
    """A knockout draw without penalties is NOT universally invalid: before
    penalty shootouts existed, drawn knockout matches were replayed instead
    (verified real case: Brazil 1-1 Czechoslovakia, 1938 quarter-final,
    replayed two days later). MatchRecord must accept this, not reject it --
    see test_knockout_matches_never_end_in_a_draw for the *current*-format
    business rule that this doesn't happen in the wc2026 dataset."""
    record = MatchRecord(
        match_id="x",
        date="1938-06-12",
        tournament="FIFA World Cup",
        season=1938,
        stage="quarter_final",
        is_knockout=True,
        home_team="Brazil",
        away_team="Czechoslovakia",
        neutral_venue=True,
        home_score=1,
        away_score=1,
        winner="draw",
        went_to_penalties=False,
        data_source="test",
    )
    assert record.winner == "draw"


def test_missing_value_report_flags_sparse_columns():
    df = pd.DataFrame({"a": [1, 2, None, None], "b": [1, 2, 3, 4]})
    report = missing_value_report(df, "test")
    a_row = report[report["column"] == "a"].iloc[0]
    assert a_row["missing_count"] == 2
    assert a_row["missing_pct"] == pytest.approx(50.0)


def test_assert_chronological_raises_on_unsorted_data():
    df = pd.DataFrame({"date": pd.to_datetime(["2026-01-02", "2026-01-01"])})
    with pytest.raises(ValueError):
        assert_chronological(df)


def test_historical_fixture_loads_against_the_verified_piterfm_schema():
    """HistoricalMatchesRawSchema was verified against a real Kaggle download of
    matches_1930_2022.csv on 2026-07-05 -- this fixture mirrors that real shape."""
    df = load_raw_historical(PITERFM_FIXTURE)
    assert len(df) > 0
    assert {"Year", "home_team", "away_team", "Round", "home_score", "away_score"}.issubset(df.columns)


def test_assert_no_future_leakage_detects_violation():
    df = pd.DataFrame(
        {
            "match_date": pd.to_datetime(["2026-06-01"]),
            "source_date": pd.to_datetime(["2026-06-02"]),
        }
    )
    with pytest.raises(ValueError):
        assert_no_future_leakage(df, as_of_col="match_date", source_date_col="source_date")
