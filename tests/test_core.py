"""Unit tests for the working core (Elo, goal model, simulation)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wc2026.config.schema import AppConfig
from wc2026.models.elo import EloEngine, expected_score
from wc2026.models.goal_model import (
    EloPoissonModel,
    outcome_probabilities,
    scoreline_matrix,
)
from wc2026.simulation.bracket import assign_teams, build_empty_r32_bracket
from wc2026.simulation.simulator import TournamentSimulator


def test_expected_score_symmetry() -> None:
    assert expected_score(1500, 1500) == pytest.approx(0.5)
    assert expected_score(1600, 1400) > 0.5
    assert expected_score(1400, 1600) < 0.5


def test_elo_zero_sum_update() -> None:
    cfg = AppConfig().elo
    eng = EloEngine(cfg)
    before = eng.rating("A") + eng.rating("B")
    eng.process_match("A", "B", 3, 0, "Friendly", neutral=True)
    after = eng.rating("A") + eng.rating("B")
    # Elo transfers points: total conserved.
    assert before == pytest.approx(after)
    assert eng.rating("A") > eng.rating("B")


def test_elo_replay_requires_sorted() -> None:
    eng = EloEngine(AppConfig().elo)
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
        eng.replay(df)


def test_scoreline_matrix_normalised() -> None:
    mat = scoreline_matrix(1.4, 1.1, max_goals=10)
    assert mat.sum() == pytest.approx(1.0)
    p_h, p_d, p_a = outcome_probabilities(mat)
    assert p_h + p_d + p_a == pytest.approx(1.0)
    assert p_h > p_a  # higher lambda side favoured


def test_goal_model_home_advantage() -> None:
    model = EloPoissonModel()
    lh_n, la_n = model.expected_goals(1500, 1500, neutral=True)
    lh_h, la_h = model.expected_goals(1500, 1500, neutral=False)
    assert lh_n == pytest.approx(la_n)  # neutral even match symmetric
    assert lh_h > lh_n  # home gets a boost


def test_simulation_probability_invariants() -> None:
    cfg = AppConfig()
    cfg.simulation.n_simulations = 2000
    teams = [f"T{i:02d}" for i in range(32)]
    ratings = {t: 1500.0 + (16 - int(t[1:])) * 5 for t in teams}
    bracket = build_empty_r32_bracket()
    assign_teams(bracket, teams)
    sim = TournamentSimulator(EloPoissonModel(), ratings, cfg.simulation, cfg.poisson)
    res = sim.run(bracket)

    assert len(res) == 32
    assert res["p_champion"].sum() == pytest.approx(1.0, abs=1e-6)
    assert res["p_final"].sum() == pytest.approx(2.0, abs=1e-6)
    assert res["p_round_of_16"].sum() == pytest.approx(32.0, abs=1e-6)
    assert (res["p_champion"] <= res["p_final"] + 1e-9).all()


def test_bracket_requires_32() -> None:
    bracket = build_empty_r32_bracket()
    with pytest.raises(ValueError, match="32"):
        assign_teams(bracket, ["only", "two"])
