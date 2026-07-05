"""Schema definitions.

Two layers:

1. Raw-file pandera schemas -- validate each source CSV exactly as it comes
   from the provider, right after download. These encode only what has been
   *verified* against the real files (see configs/data.yaml comments); if a
   provider changes a column we want a loud, specific failure here, not a
   silent KeyError three modules downstream.

2. ``MatchRecord`` -- the canonical, source-agnostic match schema every
   loader must normalize into. This is what the feature pipeline, models and
   simulation all consume, so adding a new competition (Euro, Copa America, ...)
   only means writing one more loader that emits this shape.
"""

from __future__ import annotations

import datetime as dt
from typing import Literal

import pandera.pandas as pa
from pandera.typing import Series
from pydantic import BaseModel, model_validator

# --- Raw schemas: mominullptr/FIFA-World-Cup-2026-Dataset -----------------------
# Column names verified by fetching the real CSVs from raw.githubusercontent.com.


class TeamsRawSchema(pa.DataFrameModel):
    team_id: Series[int]
    team_name: Series[str]
    fifa_code: Series[str]
    group_letter: Series[str]
    confederation: Series[str]
    fifa_ranking_pre_tournament: Series[int] = pa.Field(ge=1)
    elo_rating: Series[float]
    manager_name: Series[str] = pa.Field(nullable=True)

    class Config:
        strict = False
        coerce = True


class VenuesRawSchema(pa.DataFrameModel):
    venue_id: Series[int]
    stadium_name: Series[str] = pa.Field(nullable=True)
    city: Series[str] = pa.Field(nullable=True)
    country: Series[str] = pa.Field(nullable=True)
    capacity: Series[float] = pa.Field(nullable=True)
    latitude: Series[float] = pa.Field(nullable=True)
    longitude: Series[float] = pa.Field(nullable=True)
    elevation_meters: Series[float] = pa.Field(nullable=True)

    class Config:
        strict = False
        coerce = True


class TournamentStagesRawSchema(pa.DataFrameModel):
    stage_id: Series[int]
    stage_name: Series[str]
    is_knockout: Series[bool]

    class Config:
        strict = False
        coerce = True


class MatchesRawSchema(pa.DataFrameModel):
    match_id: Series[int]
    date: Series[str]
    kickoff_time_utc: Series[str] = pa.Field(nullable=True)
    stage_id: Series[int]
    venue_id: Series[int] = pa.Field(nullable=True)
    home_team_id: Series[int]
    away_team_id: Series[int]
    home_score: Series[float] = pa.Field(nullable=True)
    away_score: Series[float] = pa.Field(nullable=True)
    home_penalty_score: Series[float] = pa.Field(nullable=True)
    away_penalty_score: Series[float] = pa.Field(nullable=True)
    status: Series[str]
    result_type: Series[str] = pa.Field(nullable=True)
    home_xg: Series[float] = pa.Field(nullable=True)
    away_xg: Series[float] = pa.Field(nullable=True)
    referee_id: Series[float] = pa.Field(nullable=True)
    player_of_the_match_id: Series[float] = pa.Field(nullable=True)

    class Config:
        strict = False
        coerce = True


class SquadsAndPlayersRawSchema(pa.DataFrameModel):
    player_id: Series[int]
    team_id: Series[int]
    player_name: Series[str]
    position: Series[str]
    club_team: Series[str] = pa.Field(nullable=True)
    market_value_eur: Series[float] = pa.Field(nullable=True, ge=0)
    caps: Series[float] = pa.Field(nullable=True, ge=0)
    date_of_birth: Series[str] = pa.Field(nullable=True)
    height_cm: Series[float] = pa.Field(nullable=True)
    goals: Series[float] = pa.Field(nullable=True, ge=0)

    class Config:
        strict = False
        coerce = True


class MatchTeamStatsRawSchema(pa.DataFrameModel):
    match_id: Series[int]
    team_id: Series[int]
    possession_pct: Series[float] = pa.Field(nullable=True, ge=0, le=100)
    total_shots: Series[float] = pa.Field(nullable=True, ge=0)
    shots_on_target: Series[float] = pa.Field(nullable=True, ge=0)
    corners: Series[float] = pa.Field(nullable=True, ge=0)
    fouls: Series[float] = pa.Field(nullable=True, ge=0)
    offsides: Series[float] = pa.Field(nullable=True, ge=0)
    saves: Series[float] = pa.Field(nullable=True, ge=0)
    player_of_the_match: Series[str] = pa.Field(nullable=True)
    data_source: Series[str] = pa.Field(nullable=True)
    last_updated: Series[str] = pa.Field(nullable=True)

    class Config:
        strict = False
        coerce = True


# --- Historical (Kaggle piterfm) raw schema -------------------------------------
# Verified 2026-07-05 against a real download of piterfm/fifa-football-world-cup
# (file matches_1930_2022.csv, 964 rows, 1930-2022). The dataset ships 5 files in
# total (see configs/data.yaml); this schema covers the match-level file only,
# which is what the pipeline needs for Elo/backtesting. Column names keep the
# provider's original mixed case (Round/Date/Year/Host/Venue/Referee) rather
# than being silently renamed -- ``alias`` only affects matching during
# validation, pandera does not rename columns.


class HistoricalMatchesRawSchema(pa.DataFrameModel):
    home_team: Series[str]
    away_team: Series[str]
    home_score: Series[int] = pa.Field(ge=0)
    away_score: Series[int] = pa.Field(ge=0)
    home_xg: Series[float] = pa.Field(nullable=True, ge=0)
    away_xg: Series[float] = pa.Field(nullable=True, ge=0)
    home_penalty: Series[float] = pa.Field(nullable=True, ge=0)
    away_penalty: Series[float] = pa.Field(nullable=True, ge=0)
    round: Series[str] = pa.Field(alias="Round")
    date: Series[str] = pa.Field(alias="Date")
    year: Series[int] = pa.Field(alias="Year")
    host: Series[str] = pa.Field(alias="Host")
    venue: Series[str] = pa.Field(alias="Venue", nullable=True)
    referee: Series[str] = pa.Field(alias="Referee", nullable=True)
    # notes: used to detect extra time (no dedicated column exists), see preprocessing.py
    notes: Series[str] = pa.Field(alias="Notes", nullable=True)

    class Config:
        strict = False
        coerce = True


# --- Canonical match schema -----------------------------------------------------

StageName = Literal[
    "group_stage",
    "round_of_32",
    "round_of_16",
    "quarter_final",
    "semi_final",
    "third_place",
    "final",
]

WinnerSide = Literal["home", "away", "draw"]


class MatchRecord(BaseModel):
    """Source-agnostic, model-ready match row. Every loader normalizes into this."""

    match_id: str
    date: dt.date
    tournament: str
    season: int
    stage: StageName
    is_knockout: bool
    home_team: str
    away_team: str
    neutral_venue: bool
    venue: str | None = None
    stadium: str | None = None
    country: str | None = None
    venue_elevation_m: float | None = None
    home_score: int | None = None
    away_score: int | None = None
    winner: WinnerSide | None = None
    went_to_extra_time: bool = False
    went_to_penalties: bool = False
    home_penalty_score: int | None = None
    away_penalty_score: int | None = None
    home_xg: float | None = None
    away_xg: float | None = None
    home_fifa_ranking: int | None = None
    away_fifa_ranking: int | None = None
    data_source: str

    @model_validator(mode="after")
    def _check_result_consistency(self) -> MatchRecord:
        played = self.home_score is not None and self.away_score is not None
        if played and self.winner is None:
            if self.home_score == self.away_score and not self.went_to_penalties:
                object.__setattr__(self, "winner", "draw")
            elif self.went_to_penalties:
                pass  # winner must be set explicitly by the penalty resolver
            elif self.home_score is not None and self.away_score is not None:
                object.__setattr__(
                    self, "winner", "home" if self.home_score > self.away_score else "away"
                )
        # Note: a knockout match ending in "draw" here is *not* rejected. In the
        # modern format (wc2026 and recent WCs) a drawn knockout match always
        # goes to penalties, so winner is never actually "draw" in practice --
        # but pre-1970s/80s World Cups resolved drawn knockout matches with a
        # separate replay fixture instead of extra time/penalties (e.g. the
        # verified real case Brazil 1-1 Czechoslovakia, 1938 quarter-final,
        # replayed two days later). Rejecting that as invalid would be wrong:
        # it is exactly what happened. See test_schemas/test_data_validation
        # for the business-rule check that the *current* dataset never does this.
        return self
