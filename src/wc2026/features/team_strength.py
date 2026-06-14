"""Team strength aggregation.

Combines Elo ratings, rolling form, and (once available) optional richer
features into a unified per-team strength snapshot used by the dashboard
and reporting layer.

The optional features (market_value_eur, fifa_rank, squad_quality_score)
are null placeholders until the corresponding data sources are ingested —
see loaders.py TODO markers. The composite_score property degrades gracefully
to Elo-only when those fields are absent.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from wc2026.features.rolling_form import get_team_form


@dataclass
class TeamStrength:
    """Aggregated team strength at a point in time."""

    team: str
    elo: float
    ppg_5: float | None = None
    winrate_5: float | None = None
    goals_for_5: float | None = None
    goals_against_5: float | None = None
    # Optional features — null until data sources are ingested.
    market_value_eur: float | None = None  # TODO(features-mv)
    fifa_rank: int | None = None  # TODO(features-rank)
    squad_quality_score: float | None = None  # TODO(features-squad)
    extra: dict[str, float] = field(default_factory=dict)

    @property
    def composite_score(self) -> float:
        """Elo adjusted by form bonus.

        A production version would weight all available features via a fitted
        regression. For now: form above 1.5 ppg adds up to ~50 Elo points,
        below 1.5 subtracts proportionally.
        """
        score = self.elo
        if self.ppg_5 is not None:
            score += (self.ppg_5 - 1.5) * 33.0
        return score


def compute_team_strengths(
    elo_ratings: dict[str, float],
    matches: pd.DataFrame | None = None,
    as_of: pd.Timestamp | str | None = None,
    window: int = 5,
) -> dict[str, TeamStrength]:
    """Build TeamStrength objects for each team with known Elo ratings.

    Args:
        elo_ratings: Dict mapping team name to current Elo rating.
        matches: Full match frame for rolling form computation. If None,
            form metrics are left as None (Elo-only mode).
        as_of: Cut-off date for form features (default: use all matches).
        window: Rolling window for form stats.

    Returns:
        Dict mapping team name to TeamStrength.
    """
    as_of_ts = pd.Timestamp(as_of) if as_of else pd.Timestamp.max

    strengths: dict[str, TeamStrength] = {}
    for team, elo in elo_ratings.items():
        form: dict[str, float | None] = {"ppg": None, "winrate": None, "gf": None, "ga": None}
        if matches is not None:
            form = get_team_form(matches, team, as_of=as_of_ts, window=window)

        strengths[team] = TeamStrength(
            team=team,
            elo=elo,
            ppg_5=form.get("ppg"),
            winrate_5=form.get("winrate"),
            goals_for_5=form.get("gf"),
            goals_against_5=form.get("ga"),
        )

    return strengths