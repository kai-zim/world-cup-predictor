"""Poisson scoreline sampling.

``size`` is always explicit and ``lambda_home``/``lambda_away`` may be a
scalar (same expected goals for every draw -- a fixed, already-known
matchup) or a per-draw array (the matchup itself varies by simulation, as
happens once a bracket round has to be inferred rather than read from data).
One code path handles both via ``np.broadcast_to``.
"""

from __future__ import annotations

import numpy as np


def simulate_scores(
    lambda_home: float | np.ndarray,
    lambda_away: float | np.ndarray,
    size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    home_goals = rng.poisson(np.broadcast_to(lambda_home, size))
    away_goals = rng.poisson(np.broadcast_to(lambda_away, size))
    return home_goals, away_goals
