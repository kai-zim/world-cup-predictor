"""Tests for single-match knockout resolution (wc2026.models.goal_model)."""

from __future__ import annotations

import numpy as np
import pytest

from wc2026.config.schema import AppConfig
from wc2026.models.goal_model import resolve_knockout


def _rng(seed: int = 42) -> np.random.Generator:
    return np.random.default_rng(seed)


def _cfg() -> object:
    return AppConfig().poisson


def test_resolve_returns_one_of_two_teams() -> None:
    result = resolve_knockout("A", "B", 1.5, 1.0, _cfg(), _rng(), 1600.0, 1400.0)
    assert result.winner in {"A", "B"}


def test_winner_consistent_with_scoreline_when_not_shootout() -> None:
    result = resolve_knockout("A", "B", 1.5, 1.0, _cfg(), _rng(), 1600.0, 1400.0)
    if not result.decided_by_shootout:
        if result.home_goals > result.away_goals:
            assert result.winner == "A"
        elif result.away_goals > result.home_goals:
            assert result.winner == "B"


def test_shootout_only_when_level_scoreline() -> None:
    for seed in range(200):
        result = resolve_knockout("X", "Y", 1.3, 1.3, _cfg(), _rng(seed), 1500.0, 1500.0)
        if result.decided_by_shootout:
            assert result.home_goals == result.away_goals
        else:
            assert result.home_goals != result.away_goals


def test_no_draw_result_in_knockout() -> None:
    """Knockout must always produce a winner — no draws allowed."""
    for seed in range(50):
        result = resolve_knockout("X", "Y", 1.3, 1.3, _cfg(), _rng(seed), 1500.0, 1500.0)
        assert result.winner in {"X", "Y"}


def test_dominant_team_wins_majority() -> None:
    rng = _rng(0)
    wins = sum(
        resolve_knockout("Strong", "Weak", 4.0, 0.3, _cfg(), rng, 2100.0, 1000.0).winner
        == "Strong"
        for _ in range(500)
    )
    assert wins > 400


def test_goals_non_negative() -> None:
    result = resolve_knockout("A", "B", 1.5, 1.2, _cfg(), _rng(), 1500.0, 1500.0)
    assert result.home_goals >= 0
    assert result.away_goals >= 0


def test_result_fields_populated() -> None:
    result = resolve_knockout("A", "B", 1.5, 1.2, _cfg(), _rng(), 1550.0, 1480.0)
    assert result.home_team == "A"
    assert result.away_team == "B"
    assert isinstance(result.decided_by_shootout, bool)