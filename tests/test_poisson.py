"""Tests for the Poisson goal model (wc2026.models.goal_model)."""

from __future__ import annotations

import numpy as np
import pytest

from wc2026.models.goal_model import (
    EloPoissonModel,
    outcome_probabilities,
    scoreline_matrix,
)


def test_scoreline_matrix_sums_to_one() -> None:
    mat = scoreline_matrix(1.5, 1.2, max_goals=10)
    assert mat.sum() == pytest.approx(1.0)


def test_scoreline_matrix_shape() -> None:
    mat = scoreline_matrix(1.4, 1.1, max_goals=7)
    assert mat.shape == (8, 8)


def test_outcome_probabilities_sum_to_one() -> None:
    mat = scoreline_matrix(1.4, 1.1, max_goals=10)
    p_h, p_d, p_a = outcome_probabilities(mat)
    assert p_h + p_d + p_a == pytest.approx(1.0)
    assert all(0.0 <= p <= 1.0 for p in (p_h, p_d, p_a))


def test_stronger_team_has_higher_win_prob() -> None:
    model = EloPoissonModel()
    lam_h, lam_a = model.expected_goals(home_elo=1700, away_elo=1300, neutral=True)
    mat = scoreline_matrix(lam_h, lam_a, max_goals=10)
    p_h, _, p_a = outcome_probabilities(mat)
    assert p_h > p_a


def test_neutral_equal_elo_symmetric_lambda() -> None:
    model = EloPoissonModel()
    lam_h, lam_a = model.expected_goals(1500, 1500, neutral=True)
    assert lam_h == pytest.approx(lam_a)


def test_home_advantage_increases_home_lambda() -> None:
    model = EloPoissonModel()
    lam_h_n, _ = model.expected_goals(1500, 1500, neutral=True)
    lam_h_h, _ = model.expected_goals(1500, 1500, neutral=False)
    assert lam_h_h > lam_h_n


def test_lambda_non_negative_extreme_elo_diff() -> None:
    model = EloPoissonModel()
    # Very large Elo differences should never produce negative expected goals.
    lam_h, lam_a = model.expected_goals(500, 2500, neutral=True)
    assert lam_h >= 0.0
    assert lam_a >= 0.0


def test_higher_lambda_increases_win_probability() -> None:
    model = EloPoissonModel()
    lam_h_weak, lam_a_weak = model.expected_goals(1400, 1600, neutral=True)
    lam_h_strong, lam_a_strong = model.expected_goals(1600, 1400, neutral=True)
    mat_weak = scoreline_matrix(lam_h_weak, lam_a_weak, max_goals=10)
    mat_strong = scoreline_matrix(lam_h_strong, lam_a_strong, max_goals=10)
    p_h_weak, _, _ = outcome_probabilities(mat_weak)
    p_h_strong, _, _ = outcome_probabilities(mat_strong)
    assert p_h_strong > p_h_weak


def test_scoreline_matrix_non_negative_entries() -> None:
    mat = scoreline_matrix(1.3, 0.9, max_goals=10)
    assert np.all(mat >= 0.0)