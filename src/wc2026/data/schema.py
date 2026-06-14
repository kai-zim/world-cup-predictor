"""Canonical data contracts.

These dataclasses/enums define the *internal* schema every downstream module
relies on. Raw sources are messy and inconsistent; the data-loading layer is
responsible for mapping them onto these contracts so that the rest of the
codebase never sees source-specific quirks.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MatchStage(str, Enum):
    GROUP = "group"
    ROUND_OF_32 = "round_of_32"
    ROUND_OF_16 = "round_of_16"
    QUARTER_FINAL = "quarter_final"
    SEMI_FINAL = "semi_final"
    THIRD_PLACE = "third_place"
    FINAL = "final"
    OTHER = "other"  # qualifiers, friendlies, etc.


# --- Column contracts ---------------------------------------------------------
# The processed match table MUST contain at least these columns. Feature modules
# add further columns; this is the minimal backbone the Elo + simulation engine
# needs. Keeping it as a constant lets tests assert the contract explicitly.
MATCH_CORE_COLUMNS: tuple[str, ...] = (
    "date",
    "tournament",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "neutral",  # bool: True if played on neutral ground
)

# Optional feature columns. Present only when the corresponding (often
# hard-to-source) data has been ingested. Simulation gracefully degrades when
# these are absent. See features/ TODO modules for population logic.
MATCH_OPTIONAL_COLUMNS: tuple[str, ...] = (
    "home_xg",
    "away_xg",
    "home_market_value",
    "away_market_value",
    "home_fifa_rank",
    "away_fifa_rank",
    "home_rest_days",
    "away_rest_days",
    "went_to_extra_time",
    "went_to_penalties",
)


@dataclass(frozen=True, slots=True)
class TeamRating:
    """A team's state at a point in time, used by the simulation engine."""

    team: str
    elo: float
    # Optional learned attack/defence strengths (None -> fall back to Elo).
    attack_strength: float | None = None
    defence_strength: float | None = None


@dataclass(frozen=True, slots=True)
class FixtureResult:
    """Outcome of a single simulated or real match."""

    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    winner: str  # team name; ties only allowed in group stage
    decided_by_shootout: bool = False
