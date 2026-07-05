"""Knockout match resolution: regulation -> extra time -> penalties.

A knockout match never ends in a draw here, by construction: if regulation
is level, extra time is simulated at a dampened goal rate, and if still
level, a penalty shootout is resolved as a fair coin flip (see
configs/simulation.yaml -- this is a documented simplification, not a
per-team skill model for penalties, which the data does not support).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from worldcup_predictor.simulation.match_simulator import simulate_scores
from worldcup_predictor.utils.config import KnockoutConfig

EXTRA_TIME_FRACTION = 30 / 90  # extra time is 30 minutes vs. 90 for regulation


@dataclass
class KnockoutOutcome:
    home_goals: np.ndarray
    away_goals: np.ndarray
    winner_is_home: np.ndarray
    went_to_extra_time: np.ndarray
    went_to_penalties: np.ndarray


def resolve_knockout(
    lambda_home: float | np.ndarray,
    lambda_away: float | np.ndarray,
    size: int,
    rng: np.random.Generator,
    config: KnockoutConfig,
) -> KnockoutOutcome:
    home_goals, away_goals = simulate_scores(lambda_home, lambda_away, size, rng)

    draw_mask = home_goals == away_goals
    went_to_extra_time = draw_mask.copy()
    if draw_mask.any():
        et_dampening = config.extra_time_goal_rate_multiplier * EXTRA_TIME_FRACTION
        et_lambda_home = np.broadcast_to(lambda_home, size)[draw_mask] * et_dampening
        et_lambda_away = np.broadcast_to(lambda_away, size)[draw_mask] * et_dampening
        et_home, et_away = simulate_scores(et_lambda_home, et_lambda_away, int(draw_mask.sum()), rng)
        home_goals[draw_mask] += et_home
        away_goals[draw_mask] += et_away

    still_draw_mask = home_goals == away_goals
    went_to_penalties = still_draw_mask.copy()
    winner_is_home = home_goals > away_goals
    if still_draw_mask.any():
        n_shootouts = int(still_draw_mask.sum())
        penalty_home_wins = rng.random(n_shootouts) < config.penalty_shootout_home_win_prob
        winner_is_home[still_draw_mask] = penalty_home_wins

    return KnockoutOutcome(
        home_goals=home_goals,
        away_goals=away_goals,
        winner_is_home=winner_is_home,
        went_to_extra_time=went_to_extra_time,
        went_to_penalties=went_to_penalties,
    )
