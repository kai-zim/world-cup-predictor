"""Goal model + single-match resolution.

Two interchangeable goal models live behind the :class:`GoalModel` protocol:

1. :class:`EloPoissonModel` (default, fully analytic) — maps the Elo difference
   to expected goals for each side via a logistic-ish supremacy curve, then
   treats goals as independent Poisson variables. No training data required,
   which is why it is the production default given the data constraints.

2. A learned model (LightGBM/XGBoost regressor) — predicts expected goals from
   the engineered feature vector. Stubbed in models/learned.py; plugs into the
   same interface so the simulator is agnostic to which backend is active.

The scoreline distribution is the outer product of the two Poisson PMFs,
truncated at ``max_goals``. From it we derive win/draw/loss probabilities and
sample concrete scorelines during Monte-Carlo simulation.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from scipy.stats import poisson

from wc2026.config.schema import PoissonConfig
from wc2026.data.schema import FixtureResult


class GoalModel(Protocol):
    """Anything that can produce expected goals (lambda) for a fixture."""

    def expected_goals(
        self, home_elo: float, away_elo: float, neutral: bool
    ) -> tuple[float, float]:
        """Return (lambda_home, lambda_away)."""
        ...


class EloPoissonModel:
    """Analytic Elo -> expected-goals mapping.

    Calibration constants are sensible defaults from public international-match
    analyses; they should be re-fit against the training window for production
    (see evaluation/calibration.py).
    """

    def __init__(
        self,
        base_rate: float = 1.35,
        supremacy_scale: float = 0.0024,
        home_advantage_goals: float = 0.30,
    ) -> None:
        # base_rate ~ average goals per team in an evenly matched int'l match.
        self._base = base_rate
        self._scale = supremacy_scale
        self._home_goal_adv = home_advantage_goals

    def expected_goals(
        self, home_elo: float, away_elo: float, neutral: bool
    ) -> tuple[float, float]:
        diff = home_elo - away_elo
        # Supremacy translates Elo diff into a goal advantage symmetrically.
        supremacy = np.tanh(self._scale * diff)  # in (-1, 1)
        home_adv = 0.0 if neutral else self._home_goal_adv
        lam_home = max(0.05, self._base * (1.0 + supremacy) + home_adv)
        lam_away = max(0.05, self._base * (1.0 - supremacy))
        return float(lam_home), float(lam_away)


def scoreline_matrix(
    lam_home: float, lam_away: float, max_goals: int
) -> np.ndarray:
    """Joint scoreline probability matrix P[i, j] = P(home=i, away=j)."""
    home_pmf = poisson.pmf(np.arange(max_goals + 1), lam_home)
    away_pmf = poisson.pmf(np.arange(max_goals + 1), lam_away)
    mat = np.outer(home_pmf, away_pmf)
    return mat / mat.sum()  # renormalise after truncation


def outcome_probabilities(matrix: np.ndarray) -> tuple[float, float, float]:
    """Return (P_home_win, P_draw, P_away_win) from a scoreline matrix."""
    p_home = float(np.tril(matrix, -1).sum())  # home goals > away goals
    p_away = float(np.triu(matrix, 1).sum())
    p_draw = float(np.trace(matrix))
    return p_home, p_draw, p_away


def resolve_knockout(
    home_team: str,
    away_team: str,
    lam_home: float,
    lam_away: float,
    cfg: PoissonConfig,
    rng: np.random.Generator,
    home_elo: float,
    away_elo: float,
) -> FixtureResult:
    """Sample one knockout fixture (no draws allowed).

    Goals are sampled from independent Poissons. A level score is broken by a
    shootout whose win probability is nudged by the Elo difference
    (``shootout_elo_scale``); set that to 0 for a pure coin-flip.
    """
    home_goals = int(rng.poisson(lam_home))
    away_goals = int(rng.poisson(lam_away))

    if home_goals != away_goals:
        winner = home_team if home_goals > away_goals else away_team
        return FixtureResult(home_team, away_team, home_goals, away_goals, winner)

    # Shootout: logistic tilt by Elo difference.
    p_home = 1.0 / (1.0 + 10.0 ** (-(home_elo - away_elo) * cfg.shootout_elo_scale))
    winner = home_team if rng.random() < p_home else away_team
    return FixtureResult(
        home_team, away_team, home_goals, away_goals, winner, decided_by_shootout=True
    )
