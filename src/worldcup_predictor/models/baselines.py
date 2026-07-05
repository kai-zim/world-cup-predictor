"""Simple, dependency-free baseline models.

Every learned model in this project must beat these on held-out data, or
there is no point using it -- that comparison is the point of having
baselines at all, not a formality.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from scipy.stats import poisson

OUTCOME_COLUMNS = ["p_home", "p_draw", "p_away"]


class OutcomeBaseline(Protocol):
    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame: ...


def poisson_match_outcome_probs(
    lambda_home: np.ndarray | pd.Series, lambda_away: np.ndarray | pd.Series, max_goals: int = 10
) -> pd.DataFrame:
    """1x2 probabilities from independent Poisson-distributed goal counts.

    Shared by the Poisson baseline, the trained goal model and the match
    simulator's sanity checks -- one implementation of the score grid.
    """
    lambda_home = np.asarray(lambda_home, dtype=float)
    lambda_away = np.asarray(lambda_away, dtype=float)
    goals = np.arange(max_goals + 1)

    p_home = np.zeros(len(lambda_home))
    p_draw = np.zeros(len(lambda_home))
    p_away = np.zeros(len(lambda_home))

    for i in range(len(lambda_home)):
        home_pmf = poisson.pmf(goals, lambda_home[i])
        away_pmf = poisson.pmf(goals, lambda_away[i])
        grid = np.outer(home_pmf, away_pmf)
        p_home[i] = np.triu(grid, k=1).sum()
        p_draw[i] = np.trace(grid)
        p_away[i] = np.tril(grid, k=-1).sum()

    total = p_home + p_draw + p_away
    return pd.DataFrame({"p_home": p_home / total, "p_draw": p_draw / total, "p_away": p_away / total})


@dataclass
class FifaRankingBaseline:
    """Win probability from FIFA ranking difference via a logistic transform.

    A fixed, unfitted heuristic on purpose -- it should not be more
    sophisticated than "trust the official ranking", since that is exactly
    the naive baseline it represents.
    """

    scale: float = 25.0
    draw_rate: float = 0.24

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        expected_home = 1.0 / (1.0 + 10 ** (-df["ranking_diff"].astype(float) / self.scale))
        p_home = (1 - self.draw_rate) * expected_home
        p_away = (1 - self.draw_rate) * (1 - expected_home)
        return pd.DataFrame(
            {"p_home": p_home, "p_draw": self.draw_rate, "p_away": p_away}, index=df.index
        )


@dataclass
class EloDiffBaseline:
    """Win probability from Elo difference (standard logistic expectation),
    with a draw probability that shrinks as the rating gap widens -- evenly
    matched teams draw more often than lopsided ones."""

    base_draw_rate: float = 0.27
    draw_decay: float = 0.0015

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        elo_diff = df["elo_diff"].astype(float)
        expected_home = 1.0 / (1.0 + 10 ** (-elo_diff / 400.0))
        p_draw = self.base_draw_rate * np.exp(-self.draw_decay * elo_diff.abs())
        p_home = (1 - p_draw) * expected_home
        p_away = (1 - p_draw) * (1 - expected_home)
        return pd.DataFrame({"p_home": p_home, "p_draw": p_draw, "p_away": p_away}, index=df.index)


@dataclass
class HistoricalWinRateBaseline:
    """Constant-probability baseline: the observed home/draw/away frequency
    in the training set, ignoring everything about the two teams involved.
    The floor every other model must clear."""

    p_home: float = 0.45
    p_draw: float = 0.25
    p_away: float = 0.30

    @classmethod
    def fit(cls, train_df: pd.DataFrame) -> HistoricalWinRateBaseline:
        counts = train_df["winner"].value_counts(normalize=True)
        return cls(
            p_home=float(counts.get("home", 0.0)),
            p_draw=float(counts.get("draw", 0.0)),
            p_away=float(counts.get("away", 0.0)),
        )

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        n = len(df)
        return pd.DataFrame(
            {"p_home": [self.p_home] * n, "p_draw": [self.p_draw] * n, "p_away": [self.p_away] * n},
            index=df.index,
        )


@dataclass
class PoissonBaseline:
    """Expected goals from Elo difference via a simple exponential link,
    turned into 1x2 probabilities through the Poisson score grid.

    This plays the role of the spec's "Goal Model" + "Poisson Match
    Simulator" combined, in its simplest, unfitted form: no per-team
    attack/defense strength, just Elo-driven scoring rates.
    """

    base_goals: float = 1.35
    elo_scale: float = 400.0
    max_goals: int = 10

    def expected_goals(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        elo_diff = df["elo_diff"].astype(float).to_numpy()
        lambda_home = self.base_goals * np.exp(elo_diff / (2 * self.elo_scale))
        lambda_away = self.base_goals * np.exp(-elo_diff / (2 * self.elo_scale))
        return lambda_home, lambda_away

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        lambda_home, lambda_away = self.expected_goals(df)
        result = poisson_match_outcome_probs(lambda_home, lambda_away, self.max_goals)
        result.index = df.index
        return result
