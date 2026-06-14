"""Tests for xG / expected-goals interpretation via the Poisson model.

The system does not model shot-level xG (per-shot probability) for national
teams — that data is too sparse. Instead, 'expected goals' here refers to the
Poisson lambda produced by EloPoissonModel, which can be interpreted as the
per-match expected goal tally for each side. These tests verify the
mathematical properties of that model.
"""

from __future__ import annotations

import pytest

from wc2026.models.goal_model import EloPoissonModel, outcome_probabilities, scoreline_matrix


def test_expected_goals_positive() -> None:
    model = EloPoissonModel()
    lam_h, lam_a = model.expected_goals(1500, 1500, neutral=True)
    assert lam_h > 0
    assert lam_a > 0


def test_expected_goals_range_plausible() -> None:
    """Lambda should stay in a plausible international-football range."""
    model = EloPoissonModel()
    for diff in range(-500, 501, 100):
        lam_h, lam_a = model.expected_goals(1500 + diff, 1500 - diff, neutral=True)
        assert 0.05 <= lam_h <= 5.0, f"lam_h={lam_h} out of range for diff={diff}"
        assert 0.05 <= lam_a <= 5.0, f"lam_a={lam_a} out of range for diff={diff}"


def test_xg_advantage_translates_to_win_prob() -> None:
    """A team with higher xG should have a higher win probability."""
    model = EloPoissonModel()
    lam_h, lam_a = model.expected_goals(1600, 1400, neutral=True)
    assert lam_h > lam_a
    mat = scoreline_matrix(lam_h, lam_a, max_goals=10)
    p_h, _, p_a = outcome_probabilities(mat)
    assert p_h > p_a


def test_zero_elo_diff_equal_xg_neutral() -> None:
    model = EloPoissonModel()
    lam_h, lam_a = model.expected_goals(1500, 1500, neutral=True)
    assert lam_h == pytest.approx(lam_a, abs=1e-6)


def test_custom_base_rate_respected() -> None:
    low_scoring = EloPoissonModel(base_rate=0.8)
    high_scoring = EloPoissonModel(base_rate=2.0)
    lam_l, _ = low_scoring.expected_goals(1500, 1500, neutral=True)
    lam_h, _ = high_scoring.expected_goals(1500, 1500, neutral=True)
    assert lam_h > lam_l