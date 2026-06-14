"""Data loading + normalisation.

Primary realistic source: the Kaggle dataset *"International football results
from 1872 to 2024"* (martj42/international-football-results), which ships as
``results.csv`` with columns:
    date, home_team, away_team, home_score, away_score, tournament, city,
    country, neutral

This loader maps that schema onto :data:`MATCH_CORE_COLUMNS` and adds a
``stage`` classification. Other sources (Elo CSV exports, FIFA ranking dumps)
are joined in via dedicated functions, each guarded with a TODO where the data
is not freely/cleanly available.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from wc2026.data.schema import MATCH_CORE_COLUMNS, MatchStage
from wc2026.utils.logging import get_logger

log = get_logger(__name__)


# Map raw Kaggle column names -> canonical names. Kaggle already mostly matches,
# but this indirection protects us if a source renames things.
_RESULTS_COLUMN_MAP = {
    "date": "date",
    "home_team": "home_team",
    "away_team": "away_team",
    "home_score": "home_score",
    "away_score": "away_score",
    "tournament": "tournament",
    "neutral": "neutral",
}


def load_results(path: str | Path) -> pd.DataFrame:
    """Load and normalise the international results CSV.

    Returns a frame containing exactly :data:`MATCH_CORE_COLUMNS`, sorted by
    date ascending (required for the sequential Elo update).
    """
    df = pd.read_csv(path)
    missing = set(_RESULTS_COLUMN_MAP) - set(df.columns)
    if missing:
        raise ValueError(f"results file missing expected columns: {sorted(missing)}")

    df = df.rename(columns=_RESULTS_COLUMN_MAP)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df["neutral"] = df["neutral"].astype(bool)

    df = df.loc[:, list(MATCH_CORE_COLUMNS)].sort_values("date").reset_index(drop=True)
    log.info("loaded_results", rows=len(df), span=(str(df.date.min().date()), str(df.date.max().date())))
    return df


def filter_window(
    df: pd.DataFrame, start: str | pd.Timestamp, end: str | pd.Timestamp
) -> pd.DataFrame:
    """Restrict to matches in ``[start, end]`` inclusive.

    Used to build the training window: *last World Cup -> end of current group
    stage*. This is the primary leakage guard at the data level — never feed
    matches dated after the prediction horizon.
    """
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    mask = (df["date"] >= start_ts) & (df["date"] <= end_ts)
    out = df.loc[mask].reset_index(drop=True)
    log.info("filtered_window", rows=len(out), start=str(start_ts.date()), end=str(end_ts.date()))
    return out


def classify_stage(tournament: str, raw_stage: str | None = None) -> MatchStage:
    """Best-effort mapping of a match to a tournament stage.

    The Kaggle results file does NOT include stage information, so for non-WC
    matches we return OTHER. Knockout-stage labelling for the target World Cup
    comes from the dedicated bracket definition (see simulation/bracket.py),
    not from this function.
    """
    if raw_stage is None:
        return MatchStage.OTHER
    key = raw_stage.strip().lower()
    mapping = {
        "group": MatchStage.GROUP,
        "round of 32": MatchStage.ROUND_OF_32,
        "round of 16": MatchStage.ROUND_OF_16,
        "quarter-finals": MatchStage.QUARTER_FINAL,
        "semi-finals": MatchStage.SEMI_FINAL,
        "final": MatchStage.FINAL,
        "third place": MatchStage.THIRD_PLACE,
    }
    return mapping.get(key, MatchStage.OTHER)


# --- External sources: guarded stubs -----------------------------------------

def load_fifa_rankings(path: str | Path) -> pd.DataFrame:  # noqa: ARG001
    """Load historical FIFA rankings.

    TODO(data): FIFA publishes ranking snapshots but not as a clean time-series
    API. A maintained Kaggle mirror exists (cashncarry/fifaworldranking) with
    columns [rank, country_full, total_points, rank_date]. Implement the
    as-of join (rank_date <= match date) here once that source is vendored into
    data/external/. Until then, ranking-diff features stay null and the model
    backend must not require them.
    """
    raise NotImplementedError("FIFA ranking ingestion not yet implemented — see TODO(data).")


def load_market_values(path: str | Path) -> pd.DataFrame:  # noqa: ARG001
    """Load squad market values per team per tournament.

    TODO(data): Transfermarkt has no official API and scraping violates its ToS;
    historical snapshots are also not reliably retrievable. This feature is
    effectively a manual-curation task. Leave unimplemented; MarketValueDiff
    features remain null placeholders.
    """
    raise NotImplementedError("Market-value ingestion is out of scope — see TODO(data).")
