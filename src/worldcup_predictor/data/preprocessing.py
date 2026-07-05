"""Raw -> interim normalization: join the wc2026 raw tables into the
canonical, source-agnostic match schema (``MatchRecord``).

Team-name normalization, date parsing, match-ID creation and stage-name
unification all happen here, once, so every downstream module (features,
models, simulation) works off one consistent shape regardless of which raw
source a match came from.
"""

from __future__ import annotations

import pandas as pd

from worldcup_predictor.data.schemas import MatchRecord
from worldcup_predictor.data.validation import missing_value_report
from worldcup_predictor.utils.logging import get_logger

logger = get_logger(__name__)

# Verified against tournament_stages.csv (see plan research notes).
_STAGE_NAME_MAP = {
    "Group Stage": "group_stage",
    "Round of 32": "round_of_32",
    "Round of 16": "round_of_16",
    "Quarter-finals": "quarter_final",
    "Semi-finals": "semi_final",
    "Third-place match": "third_place",
    "Final": "final",
}

CANONICAL_COLUMNS = [
    "match_id",
    "date",
    "tournament",
    "season",
    "stage",
    "is_knockout",
    "home_team",
    "away_team",
    "neutral_venue",
    "venue",
    "stadium",
    "country",
    "venue_elevation_m",
    "home_score",
    "away_score",
    "winner",
    "went_to_extra_time",
    "went_to_penalties",
    "home_penalty_score",
    "away_penalty_score",
    "home_xg",
    "away_xg",
    "home_fifa_ranking",
    "away_fifa_ranking",
    "data_source",
]


def normalize_team_name(name: str) -> str:
    """Whitespace/case normalization applied to every team name, regardless of source."""
    return " ".join(str(name).strip().split())


# Aligns the historical (piterfm) dataset's naming with the wc2026 dataset's,
# verified 2026-07-05 against a real download of both. Deliberately does NOT
# merge "Czechoslovakia" into "Czechia" -- those are different national-team
# eras/associations, not a simple renaming, so Elo/form continuity should not
# be assumed across that boundary.
HISTORICAL_TEAM_ALIASES = {
    "United States": "USA",
    "Korea Republic": "South Korea",
    "Czech Republic": "Czechia",
}


def normalize_historical_team_name(name: str) -> str:
    normalized = normalize_team_name(name)
    return HISTORICAL_TEAM_ALIASES.get(normalized, normalized)


def _validate_and_finalize(df: pd.DataFrame, id_label: str) -> pd.DataFrame:
    """Shared tail of both builders: MatchRecord-validate every row (loud,
    specific failures instead of silent NaNs downstream), then sort by date."""
    validated_rows = []
    for raw_row in df[CANONICAL_COLUMNS].to_dict(orient="records"):
        row = {k: (None if pd.isna(v) else v) for k, v in raw_row.items()}
        try:
            record = MatchRecord.model_validate(row)
        except Exception as exc:
            raise ValueError(
                f"{id_label} {row.get('match_id')} ({row.get('home_team')} vs "
                f"{row.get('away_team')}, season {row.get('season')}) failed "
                f"MatchRecord validation: {exc}"
            ) from exc
        validated_rows.append(record.model_dump())

    result = pd.DataFrame(validated_rows)
    result["date"] = pd.to_datetime(result["date"])
    return result.sort_values("date", kind="stable").reset_index(drop=True)


def build_match_table(
    raw: dict[str, pd.DataFrame], tournament: str = "FIFA World Cup 2026"
) -> pd.DataFrame:
    """Join wc2026 raw tables into the canonical match frame (interim data).

    ``raw`` is the dict returned by ``loaders.load_raw_wc2026`` (already
    schema-validated). Every row of the result is additionally validated
    through the ``MatchRecord`` pydantic model, so malformed joins fail loudly
    here instead of surfacing as confusing NaNs three modules downstream.
    """
    matches = raw["matches"].copy()
    teams = raw["teams"].set_index("team_id")
    venues = raw["venues"].set_index("venue_id")
    stages = raw["tournament_stages"].set_index("stage_id")

    missing_value_report(matches, "matches.csv")

    df = (
        matches.merge(
            teams[["team_name", "fifa_code", "fifa_ranking_pre_tournament"]].add_prefix("home_"),
            left_on="home_team_id",
            right_index=True,
            how="left",
        )
        .merge(
            teams[["team_name", "fifa_code", "fifa_ranking_pre_tournament"]].add_prefix("away_"),
            left_on="away_team_id",
            right_index=True,
            how="left",
        )
        .merge(
            venues[["stadium_name", "city", "country", "elevation_meters"]],
            left_on="venue_id",
            right_index=True,
            how="left",
        )
        .merge(
            stages[["stage_name", "is_knockout"]],
            left_on="stage_id",
            right_index=True,
            how="left",
        )
    )

    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["stage"] = df["stage_name"].map(_STAGE_NAME_MAP)
    unmapped = df.loc[df["stage"].isna(), "stage_name"].unique()
    if len(unmapped):
        raise ValueError(
            f"Unrecognized stage name(s) from tournament_stages.csv: {list(unmapped)}. "
            "Extend _STAGE_NAME_MAP in preprocessing.py to match the upstream data."
        )

    df["home_team"] = df["home_team_name"].map(normalize_team_name)
    df["away_team"] = df["away_team_name"].map(normalize_team_name)

    is_completed = df["status"].str.lower() == "completed"
    df["home_score"] = df["home_score"].where(is_completed)
    df["away_score"] = df["away_score"].where(is_completed)

    df["went_to_penalties"] = df["home_penalty_score"].notna() & df["away_penalty_score"].notna()
    df["went_to_extra_time"] = (
        df["result_type"].isin(["Extra Time", "Penalties"]) | df["went_to_penalties"]
    )

    df["winner"] = None
    home_win = is_completed & (df["home_score"] > df["away_score"])
    away_win = is_completed & (df["home_score"] < df["away_score"])
    draw = is_completed & (df["home_score"] == df["away_score"]) & ~df["went_to_penalties"]
    df.loc[home_win, "winner"] = "home"
    df.loc[away_win, "winner"] = "away"
    df.loc[draw, "winner"] = "draw"
    home_pk_win = (
        is_completed & df["went_to_penalties"] & (df["home_penalty_score"] > df["away_penalty_score"])
    )
    away_pk_win = (
        is_completed & df["went_to_penalties"] & (df["home_penalty_score"] < df["away_penalty_score"])
    )
    df.loc[home_pk_win, "winner"] = "home"
    df.loc[away_pk_win, "winner"] = "away"

    # A match is only "neutral" if neither side's own federation matches the host
    # country of the venue -- otherwise the home/away labels here are just fixture
    # slots (FIFA does not designate a "home team" in this format), not real home
    # advantage. USA/Mexico/Canada teams playing in their own country are the
    # documented exception.
    df["neutral_venue"] = ~(
        (df["home_fifa_code"] == df["country"]) | (df["away_fifa_code"] == df["country"])
    )

    df["match_id"] = df["match_id"].astype(str)
    df["season"] = 2026
    df["tournament"] = tournament
    df["data_source"] = "wc2026"

    df = df.rename(
        columns={
            "city": "venue",
            "stadium_name": "stadium",
            "elevation_meters": "venue_elevation_m",
            "home_fifa_ranking_pre_tournament": "home_fifa_ranking",
            "away_fifa_ranking_pre_tournament": "away_fifa_ranking",
        }
    )

    return _validate_and_finalize(df, id_label="Match")


# --- Historical (piterfm) raw -> interim -----------------------------------------
# Verified 2026-07-05 against a real download of matches_1930_2022.csv (964 rows).
# Round values cover both the modern taxonomy (Round of 16, Quarter-finals, ...)
# and older formats (two round-robin "group" rounds used 1950-1982) -- those
# older rounds are all treated as group_stage/non-knockout, which is correct:
# they were round-robin, not sudden-death, elimination.
HISTORICAL_STAGE_MAP = {
    "Group stage": "group_stage",
    "First round": "group_stage",
    "Second round": "group_stage",
    "First group stage": "group_stage",
    "Second group stage": "group_stage",
    "Group stage play-off": "group_stage",
    "Final stage": "group_stage",
    "Round of 16": "round_of_16",
    "Quarter-finals": "quarter_final",
    "Semi-finals": "semi_final",
    "Third-place match": "third_place",
    "Final": "final",
}


def _parse_host_set(host: str) -> set[str]:
    """"Korea Republic, Japan" (2002 co-host) -> {"South Korea", "Japan"}."""
    return {normalize_historical_team_name(part) for part in host.split(",")}


def build_historical_match_table(
    historical: pd.DataFrame, tournament: str = "FIFA World Cup"
) -> pd.DataFrame:
    """Normalize the piterfm historical dataset (matches_1930_2022.csv, already
    schema-validated by ``data.loaders.load_raw_historical``) into the same
    canonical match schema ``build_match_table`` produces for wc2026.

    Only Elo/rolling-form-relevant fields are populated with real values;
    squad market value and per-match FIFA ranking are not available this far
    back and stay ``None`` (see ``home_fifa_ranking``/``away_fifa_ranking``
    and README limitations) -- ``features.feature_pipeline`` has a dedicated
    ``build_historical_feature_frame`` that only computes what's actually
    available here.
    """
    df = historical.copy()

    df["stage"] = df["Round"].map(HISTORICAL_STAGE_MAP)
    unmapped = df.loc[df["stage"].isna(), "Round"].unique()
    if len(unmapped):
        raise ValueError(
            f"Unrecognized Round value(s) in historical data: {list(unmapped)}. "
            "Extend HISTORICAL_STAGE_MAP in preprocessing.py to match the upstream data."
        )
    df["is_knockout"] = df["stage"] != "group_stage"

    df["home_team"] = df["home_team"].map(normalize_historical_team_name)
    df["away_team"] = df["away_team"].map(normalize_historical_team_name)
    df["date"] = pd.to_datetime(df["Date"]).dt.date

    df["went_to_penalties"] = df["home_penalty"].notna() & df["away_penalty"].notna()
    df["went_to_extra_time"] = (
        df["Notes"].astype(str).str.contains("extra time", case=False, na=False) | df["went_to_penalties"]
    )

    df["winner"] = None
    home_win = df["home_score"] > df["away_score"]
    away_win = df["home_score"] < df["away_score"]
    draw = (df["home_score"] == df["away_score"]) & ~df["went_to_penalties"]
    df.loc[home_win, "winner"] = "home"
    df.loc[away_win, "winner"] = "away"
    df.loc[draw, "winner"] = "draw"
    home_pk_win = df["went_to_penalties"] & (df["home_penalty"] > df["away_penalty"])
    away_pk_win = df["went_to_penalties"] & (df["home_penalty"] < df["away_penalty"])
    df.loc[home_pk_win, "winner"] = "home"
    df.loc[away_pk_win, "winner"] = "away"

    # A team only has real home advantage if it's (one of) the host nation(s);
    # everyone else played on neutral ground regardless of the fixture's
    # arbitrary home/away labelling.
    host_sets = df["Host"].map(_parse_host_set)
    df["neutral_venue"] = [
        home not in hosts and away not in hosts
        for home, away, hosts in zip(df["home_team"], df["away_team"], host_sets, strict=True)
    ]

    df["match_id"] = "hist_" + df.index.astype(str)
    df["season"] = df["Year"].astype(int)
    df["tournament"] = tournament
    df["stadium"] = df["Venue"]
    df["venue"] = None
    df["country"] = df["Host"]
    df["venue_elevation_m"] = None
    df["home_fifa_ranking"] = None
    df["away_fifa_ranking"] = None
    df["home_penalty_score"] = df["home_penalty"]
    df["away_penalty_score"] = df["away_penalty"]
    df["data_source"] = "historical_piterfm"

    return _validate_and_finalize(df, id_label="Historical match")
