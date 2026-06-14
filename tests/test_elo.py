"""Tests for the Elo rating engine (wc2026.models.elo)."""

from __future__ import annotations

import pandas as pd
import pytest

from wc2026.config.schema import AppConfig
from wc2026.models.elo import EloEngine, expected_score


def test_expected_score_equal_ratings() -> None:
    assert expected_score(1500, 1500) == pytest.approx(0.5)


def test_expected_score_higher_beats_lower() -> None:
    assert expected_score(1600, 1400) > 0.5
    assert expected_score(1400, 1600) < 0.5


def test_expected_score_complement() -> None:
    p = expected_score(1700, 1300)
    q = expected_score(1300, 1700)
    assert p + q == pytest.approx(1.0)


def test_elo_zero_sum() -> None:
    cfg = AppConfig().elo
    eng = EloEngine(cfg)
    r_home, r_away = eng.process_match("A", "B", 2, 1, "Friendly", neutral=True)
    assert eng.rating("A") + eng.rating("B") == pytest.approx(r_home + r_away)


def test_elo_winner_gains_points() -> None:
    cfg = AppConfig().elo
    eng = EloEngine(cfg)
    eng.process_match("A", "B", 1, 0, "Friendly", neutral=True)
    assert eng.rating("A") > 1500
    assert eng.rating("B") < 1500


def test_elo_draw_from_equal_ratings_no_change() -> None:
    cfg = AppConfig().elo
    eng = EloEngine(cfg)
    eng.process_match("A", "B", 1, 1, "Friendly", neutral=True)
    # When ratings are equal a draw is the expected result -> zero update.
    assert eng.rating("A") == pytest.approx(1500)
    assert eng.rating("B") == pytest.approx(1500)


def test_elo_home_advantage_reduces_gain_on_draw() -> None:
    """Home team drawing on home ground is below expectation -> loses points."""
    cfg = AppConfig().elo
    eng_home = EloEngine(cfg)
    eng_home.process_match("A", "B", 1, 1, "Friendly", neutral=False)

    eng_neutral = EloEngine(cfg)
    eng_neutral.process_match("A", "B", 1, 1, "Friendly", neutral=True)

    assert eng_home.rating("A") < eng_neutral.rating("A")


def test_elo_unseen_team_returns_default() -> None:
    eng = EloEngine(AppConfig().elo)
    assert eng.rating("NewTeam") == AppConfig().elo.start_rating


def test_elo_replay_adds_pre_match_columns() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "tournament": ["Friendly"] * 2,
            "home_team": ["A", "C"],
            "away_team": ["B", "D"],
            "home_score": [1, 0],
            "away_score": [0, 2],
            "neutral": [True, True],
        }
    )
    eng = EloEngine(AppConfig().elo)
    out = eng.replay(df)
    assert "home_elo_pre" in out.columns
    assert "away_elo_pre" in out.columns
    # First match: both teams are new, so pre-match ratings are the default.
    assert out["home_elo_pre"].iloc[0] == pytest.approx(1500.0)
    assert out["away_elo_pre"].iloc[0] == pytest.approx(1500.0)


def test_elo_replay_requires_sorted() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-01"]),
            "tournament": ["Friendly"] * 2,
            "home_team": ["A", "C"],
            "away_team": ["B", "D"],
            "home_score": [1, 2],
            "away_score": [0, 1],
            "neutral": [True, True],
        }
    )
    with pytest.raises(ValueError, match="sorted"):
        EloEngine(AppConfig().elo).replay(df)


def test_elo_high_importance_match_produces_larger_delta() -> None:
    cfg = AppConfig().elo
    eng_wc = EloEngine(cfg)
    eng_wc.process_match("A", "B", 2, 0, "FIFA World Cup", neutral=True)

    eng_fr = EloEngine(cfg)
    eng_fr.process_match("A", "B", 2, 0, "Friendly", neutral=True)

    # World Cup match has higher K -> larger Elo swing.
    assert eng_wc.rating("A") > eng_fr.rating("A")


def test_elo_ratings_snapshot_is_copy() -> None:
    eng = EloEngine(AppConfig().elo)
    eng.process_match("A", "B", 1, 0, "Friendly", neutral=True)
    snap = eng.ratings_snapshot()
    snap["A"] = 9999.0
    assert eng.rating("A") != 9999.0