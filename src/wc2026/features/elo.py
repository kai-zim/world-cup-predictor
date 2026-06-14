"""Elo feature helpers for the feature engineering pipeline.

The core Elo engine lives in models.elo to avoid circular imports; this
module wraps it with feature-pipeline-oriented convenience functions used
by train.py and the prediction CLI.
"""

from __future__ import annotations

import pandas as pd

from wc2026.config.schema import EloConfig
from wc2026.models.elo import EloEngine
from wc2026.models.elo import expected_score as _expected_score  # re-exported below


def add_elo_features(matches: pd.DataFrame, config: EloConfig) -> pd.DataFrame:
    """Replay Elo chronologically and return the frame with pre-match Elo columns.

    Thin convenience wrapper around EloEngine.replay used by the feature
    pipeline so callers do not need to instantiate the engine directly.

    Added columns:
        home_elo_pre  — home team's Elo immediately before each match
        away_elo_pre  — away team's Elo immediately before each match
    """
    engine = EloEngine(config)
    return engine.replay(matches)


def elo_win_probability(
    home_elo: float, away_elo: float, home_advantage: float = 65.0, neutral: bool = False
) -> float:
    """Expected win probability for the home side including optional home advantage.

    Args:
        home_elo: Current Elo rating of the home (or nominally 'home') team.
        away_elo: Current Elo rating of the away team.
        home_advantage: Elo-point boost applied to home team (0 if neutral venue).
        neutral: If True, overrides home_advantage to zero.
    """
    adv = 0.0 if neutral else home_advantage
    return _expected_score(home_elo + adv, away_elo)


# Re-export so downstream modules can import from one place.
expected_score = _expected_score