"""Tests for the Monte-Carlo tournament simulator (wc2026.simulation)."""

from __future__ import annotations

import pytest

from wc2026.config.schema import AppConfig
from wc2026.models.goal_model import EloPoissonModel
from wc2026.simulation.bracket import assign_teams, build_empty_r32_bracket
from wc2026.simulation.simulator import TournamentSimulator


def _teams() -> list[str]:
    return [f"T{i:02d}" for i in range(32)]


def _make_sim(n: int = 500) -> TournamentSimulator:
    cfg = AppConfig()
    cfg.simulation.n_simulations = n
    return TournamentSimulator(EloPoissonModel(), {t: 1500.0 for t in _teams()}, cfg.simulation, cfg.poisson)


def test_champion_probabilities_sum_to_one() -> None:
    sim = _make_sim()
    bracket = build_empty_r32_bracket()
    assign_teams(bracket, _teams())
    result = sim.run(bracket)
    assert result["p_champion"].sum() == pytest.approx(1.0, abs=1e-6)


def test_all_32_teams_in_results() -> None:
    sim = _make_sim()
    bracket = build_empty_r32_bracket()
    assign_teams(bracket, _teams())
    result = sim.run(bracket)
    assert len(result) == 32


def test_monotonicity_champion_le_final() -> None:
    sim = _make_sim(n=1000)
    bracket = build_empty_r32_bracket()
    assign_teams(bracket, _teams())
    result = sim.run(bracket)
    assert (result["p_champion"] <= result["p_final"] + 1e-9).all()


def test_monotonicity_final_le_semi() -> None:
    sim = _make_sim(n=1000)
    bracket = build_empty_r32_bracket()
    assign_teams(bracket, _teams())
    result = sim.run(bracket)
    assert (result["p_final"] <= result["p_semi_final"] + 1e-9).all()


def test_higher_elo_team_wins_champion_more_often() -> None:
    cfg = AppConfig()
    cfg.simulation.n_simulations = 3000
    teams = _teams()
    # T00 gets the highest rating, T31 the lowest.
    ratings = {t: 1500.0 + (31 - int(t[1:])) * 15 for t in teams}
    bracket = build_empty_r32_bracket()
    assign_teams(bracket, teams)
    sim = TournamentSimulator(EloPoissonModel(), ratings, cfg.simulation, cfg.poisson)
    result = sim.run(bracket).set_index("team")
    assert result.loc["T00", "p_champion"] > result.loc["T31", "p_champion"]


def test_bracket_unassigned_raises() -> None:
    sim = _make_sim(n=10)
    bracket = build_empty_r32_bracket()  # leaves not assigned
    with pytest.raises(AssertionError, match="bracket not fully assigned"):
        sim.run(bracket)


def test_reproducible_with_same_seed() -> None:
    cfg = AppConfig()
    cfg.simulation.n_simulations = 500
    teams = _teams()
    ratings = {t: 1500.0 for t in teams}
    bracket = build_empty_r32_bracket()
    assign_teams(bracket, teams)

    sim1 = TournamentSimulator(EloPoissonModel(), ratings, cfg.simulation, cfg.poisson)
    sim2 = TournamentSimulator(EloPoissonModel(), ratings, cfg.simulation, cfg.poisson)
    r1 = sim1.run(bracket)
    assign_teams(bracket, teams)  # re-assign since run doesn't mutate bracket structure
    r2 = sim2.run(bracket)
    # Same seed -> same champion probabilities.
    assert list(r1["p_champion"]) == list(r2["p_champion"])