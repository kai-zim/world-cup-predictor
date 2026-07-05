import pandas as pd
import pytest

from worldcup_predictor.features.elo import compute_elo_ratings
from worldcup_predictor.utils.config import EloConfig


def _config(**overrides):
    base = dict(
        initial_rating=1500.0,
        k_factor_group_stage=30.0,
        k_factor_knockout=45.0,
        k_factor_friendly=20.0,
        home_advantage=0.0,
    )
    base.update(overrides)
    return EloConfig(**base)


def _one_match(**overrides):
    row = dict(
        match_id="1",
        date=pd.Timestamp("2026-06-01"),
        home_team="A",
        away_team="B",
        is_knockout=False,
        neutral_venue=True,
        home_score=1,
        away_score=0,
        winner="home",
    )
    row.update(overrides)
    return pd.DataFrame([row])


def test_equal_ratings_neutral_match_has_50_50_expectation():
    result = compute_elo_ratings(_one_match(), {"A": 1500.0, "B": 1500.0}, _config())
    row = result.iloc[0]
    assert row["elo_expected_home_win_prob"] == pytest.approx(0.5)
    assert row["home_elo_post"] == pytest.approx(1515.0)  # 1500 + 30 * (1 - 0.5)
    assert row["away_elo_post"] == pytest.approx(1485.0)


def test_unplayed_match_does_not_change_ratings():
    matches = _one_match(home_score=None, away_score=None, winner=None)
    result = compute_elo_ratings(matches, {"A": 1500.0, "B": 1500.0}, _config())
    row = result.iloc[0]
    assert row["home_elo_post"] == row["home_elo_pre"] == 1500.0
    assert row["away_elo_post"] == row["away_elo_pre"] == 1500.0


def test_home_advantage_increases_expected_home_win_probability():
    matches = _one_match(neutral_venue=False, home_score=None, away_score=None, winner=None)
    result = compute_elo_ratings(matches, {"A": 1500.0, "B": 1500.0}, _config(home_advantage=60.0))
    assert result.iloc[0]["elo_expected_home_win_prob"] > 0.5


def test_knockout_matches_use_a_higher_k_factor_than_group_stage():
    def home_post(is_knockout: bool) -> float:
        matches = _one_match(is_knockout=is_knockout)
        return compute_elo_ratings(matches, {"A": 1500.0, "B": 1500.0}, _config()).iloc[0]["home_elo_post"]

    assert home_post(True) > home_post(False)


def test_ratings_carry_forward_chronologically_across_matches():
    common = dict(winner="home", home_score=1, away_score=0)
    match_1 = _one_match(match_id="1", date=pd.Timestamp("2026-06-01"), **common)
    match_2 = _one_match(match_id="2", date=pd.Timestamp("2026-06-05"), **common)
    matches = pd.concat([match_1, match_2], ignore_index=True)
    result = compute_elo_ratings(matches, {"A": 1500.0, "B": 1500.0}, _config())
    # Second match's pre-rating must equal the first match's post-rating.
    assert result.iloc[1]["home_elo_pre"] == pytest.approx(result.iloc[0]["home_elo_post"])
    assert result.iloc[1]["away_elo_pre"] == pytest.approx(result.iloc[0]["away_elo_post"])
