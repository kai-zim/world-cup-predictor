from world_cup_predictor.simulation.monte_carlo import Matchup, simulate_knockout_tournament


def test_simulate_knockout_tournament_returns_empty_when_no_matchups() -> None:
    assert simulate_knockout_tournament([], n_simulations=10) == {}


def test_simulate_knockout_tournament_uses_bracket_progression() -> None:
    matchups = [
        Matchup(home_team="A", away_team="B", home_win_probability=1.0),
        Matchup(home_team="C", away_team="D", home_win_probability=1.0),
    ]
    probabilities = simulate_knockout_tournament(matchups, n_simulations=100)
    assert set(probabilities).issubset({"A", "C"})
