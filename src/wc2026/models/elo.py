"""International Elo rating engine.

Implements the World-Football-Elo update rule with tournament-importance and
goal-difference weighting. The engine processes matches strictly in
chronological order and exposes, for every match, the *pre-match* ratings of
both teams — which is exactly what downstream features may consume without
leaking the result.

Reference rule (eloratings.net):
    R_new = R_old + K * G * (W - W_e)
where
    K  = importance weight (per tournament)
    G  = goal-difference multiplier
    W  = actual result (1 win / 0.5 draw / 0 loss)
    W_e= expected result = 1 / (1 + 10^(-dr/400)),  dr = elo diff incl. home adv.
"""

from __future__ import annotations

import pandas as pd

from wc2026.config.schema import EloConfig
from wc2026.utils.logging import get_logger

log = get_logger(__name__)


def expected_score(elo_a: float, elo_b: float) -> float:
    """Expected score for A vs B given (home-adjusted) Elo difference."""
    return 1.0 / (1.0 + 10.0 ** (-(elo_a - elo_b) / 400.0))


def _goal_diff_multiplier(goal_diff: int) -> float:
    """Standard eloratings.net goal-difference weighting."""
    gd = abs(goal_diff)
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11.0 + gd) / 8.0


class EloEngine:
    """Stateful, chronological Elo rating tracker."""

    def __init__(self, config: EloConfig) -> None:
        self._cfg = config
        self._ratings: dict[str, float] = {}

    def rating(self, team: str) -> float:
        """Current rating for ``team`` (start rating if unseen)."""
        return self._ratings.get(team, self._cfg.start_rating)

    def ratings_snapshot(self) -> dict[str, float]:
        """Copy of all current ratings."""
        return dict(self._ratings)

    def _k(self, tournament: str) -> float:
        return self._cfg.importance.get(tournament, self._cfg.default_importance)

    def process_match(
        self,
        home_team: str,
        away_team: str,
        home_score: int,
        away_score: int,
        tournament: str,
        neutral: bool,
    ) -> tuple[float, float]:
        """Update ratings for one match; return the PRE-match (home, away) Elo.

        Returning pre-match ratings makes it trivial to build a leakage-free
        feature column while replaying history.
        """
        r_home = self.rating(home_team)
        r_away = self.rating(away_team)

        home_adv = 0.0 if neutral else self._cfg.home_advantage
        we_home = expected_score(r_home + home_adv, r_away)

        if home_score > away_score:
            w_home = 1.0
        elif home_score < away_score:
            w_home = 0.0
        else:
            w_home = 0.5

        k = self._k(tournament)
        g = _goal_diff_multiplier(home_score - away_score)
        delta = k * g * (w_home - we_home)

        self._ratings[home_team] = r_home + delta
        self._ratings[away_team] = r_away - delta
        return r_home, r_away

    def replay(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Replay a chronologically-sorted match frame.

        Returns the input frame with two added columns ``home_elo_pre`` and
        ``away_elo_pre`` containing pre-match ratings. The internal state ends
        at the ratings *after* the final match — ready to seed simulation.
        """
        if not matches["date"].is_monotonic_increasing:
            raise ValueError("matches must be sorted by date ascending before replay")

        home_pre: list[float] = []
        away_pre: list[float] = []
        for row in matches.itertuples(index=False):
            rh, ra = self.process_match(
                home_team=row.home_team,
                away_team=row.away_team,
                home_score=int(row.home_score),
                away_score=int(row.away_score),
                tournament=row.tournament,
                neutral=bool(row.neutral),
            )
            home_pre.append(rh)
            away_pre.append(ra)

        out = matches.copy()
        out["home_elo_pre"] = home_pre
        out["away_elo_pre"] = away_pre
        log.info("elo_replay_done", matches=len(out), teams=len(self._ratings))
        return out
