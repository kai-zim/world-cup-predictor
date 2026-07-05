import numpy as np
import pytest

from worldcup_predictor.models.train import train_goal_model
from worldcup_predictor.simulation.knockout import resolve_knockout
from worldcup_predictor.simulation.match_simulator import simulate_scores
from worldcup_predictor.simulation.tournament import simulate_tournament
from worldcup_predictor.utils.config import KnockoutConfig


def _knockout_config(**overrides):
    base = dict(penalty_shootout_home_win_prob=0.5, extra_time_goal_rate_multiplier=0.85)
    base.update(overrides)
    return KnockoutConfig(**base)


def test_simulate_scores_are_nonnegative_integers():
    rng = np.random.default_rng(0)
    home, away = simulate_scores(1.5, 1.2, 10000, rng)
    assert (home >= 0).all()
    assert (away >= 0).all()
    assert np.issubdtype(home.dtype, np.integer)


def test_resolve_knockout_always_has_a_decisive_winner():
    """A knockout match never ends undecided: either the scoreline is
    decisive, or (a tied scoreline after extra time) the shootout is. A tied
    scoreline itself is legitimate here -- penalty goals are tracked
    separately from the match score, matching real football scorekeeping."""
    rng = np.random.default_rng(0)
    outcome = resolve_knockout(1.1, 1.1, 20000, rng, _knockout_config())
    decisive_by_goals = outcome.home_goals != outcome.away_goals
    assert (decisive_by_goals | outcome.went_to_penalties).all()
    assert (outcome.went_to_penalties == (outcome.home_goals == outcome.away_goals)).all()


def test_resolve_knockout_penalty_shootout_is_roughly_fair():
    rng = np.random.default_rng(0)
    # Very low, equal lambdas -> many 0-0s -> many shootouts.
    outcome = resolve_knockout(0.05, 0.05, 50000, rng, _knockout_config())
    assert outcome.went_to_penalties.sum() > 1000
    home_win_rate_in_shootouts = outcome.winner_is_home[outcome.went_to_penalties].mean()
    assert home_win_rate_in_shootouts == pytest.approx(0.5, abs=0.05)


def test_simulate_tournament_champion_probabilities_sum_to_one(
    matches, feature_frame, played_matches, model_config, simulation_config
):
    goal_model = train_goal_model(played_matches, model_config.goal_model)
    result = simulate_tournament(matches, feature_frame, goal_model, simulation_config)
    table = result.to_probability_table()
    assert table["prob_champion"].sum() == pytest.approx(1.0, abs=1e-9)


def test_simulate_tournament_respects_already_completed_results(
    matches, feature_frame, played_matches, model_config, simulation_config
):
    goal_model = train_goal_model(played_matches, model_config.goal_model)
    result = simulate_tournament(matches, feature_frame, goal_model, simulation_config)
    table = result.to_probability_table().set_index("team")

    # Alpha and Epsilon won their (already completed) semifinals -> must reach
    # the final in every single simulation draw.
    assert table.loc["Alpha", "prob_final"] == pytest.approx(1.0)
    assert table.loc["Epsilon", "prob_final"] == pytest.approx(1.0)

    # Beta and Zeta lost their semifinals -> must never reach the final.
    assert table.loc["Beta", "prob_final"] == 0.0
    assert table.loc["Zeta", "prob_final"] == 0.0

    # Teams eliminated in the group stage never even reach the semifinal.
    for team in ["Gamma", "Delta", "Eta", "Theta"]:
        assert table.loc[team, "prob_semi_final"] == 0.0


def test_simulate_tournament_final_is_a_genuine_coin_toss_between_close_teams(
    matches, feature_frame, played_matches, model_config, simulation_config
):
    """Alpha and Epsilon enter the final with nearly identical Elo -- the
    simulated final must actually be stochastic, not a bug that hands the
    title to one side 100% of the time (see history: this exact class of bug
    was caught once already, from a None/NaN mixup silently fixing every
    unplayed match's "winner")."""
    goal_model = train_goal_model(played_matches, model_config.goal_model)
    result = simulate_tournament(matches, feature_frame, goal_model, simulation_config)
    table = result.to_probability_table().set_index("team")
    assert table.loc["Alpha", "prob_champion"] == pytest.approx(0.5, abs=0.05)
    assert table.loc["Epsilon", "prob_champion"] == pytest.approx(0.5, abs=0.05)
