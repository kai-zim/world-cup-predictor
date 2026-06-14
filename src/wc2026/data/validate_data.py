"""Data validation helpers.

Validates raw and processed DataFrames against the canonical schema before
they enter the Elo engine or feature pipeline. Raises informative errors
rather than letting silent type mismatches propagate downstream.
"""

from __future__ import annotations

import pandas as pd

from wc2026.data.schema import MATCH_CORE_COLUMNS


def validate_match_frame(df: pd.DataFrame, *, require_sorted: bool = True) -> pd.DataFrame:
    """Validate a match DataFrame against the core schema contract.

    Checks column presence, non-null critical fields, correct date dtype, and
    optionally that the frame is sorted by date ascending (required for the
    sequential Elo update to be leakage-free).

    Returns the validated frame unchanged so calls can be chained::

        df = validate_match_frame(load_results(path))
        engine.replay(df)
    """
    missing = set(MATCH_CORE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Match frame missing required columns: {sorted(missing)}")

    null_cols = [c for c in ("date", "home_team", "away_team") if df[c].isna().any()]
    if null_cols:
        raise ValueError(f"Null values found in critical columns: {null_cols}")

    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        raise ValueError("Column 'date' must be datetime64; call pd.to_datetime() before validating.")

    if df[["home_score", "away_score"]].lt(0).any(axis=None):
        raise ValueError("Negative goal scores detected — check raw data for parsing errors.")

    if require_sorted and not df["date"].is_monotonic_increasing:
        raise ValueError(
            "Match frame must be sorted by date ascending for leakage-safe Elo replay. "
            "Call .sort_values('date').reset_index(drop=True) first."
        )

    return df


def validate_seeding(teams: list[str]) -> list[str]:
    """Validate a 32-team knockout seeding list.

    Checks length, uniqueness, and that no names are empty or whitespace-only.
    Returns the list unchanged so calls can be chained.
    """
    if len(teams) != 32:
        raise ValueError(f"Seeding must contain exactly 32 teams, got {len(teams)}.")

    if len(set(teams)) != 32:
        seen: dict[str, int] = {}
        for t in teams:
            seen[t] = seen.get(t, 0) + 1
        duplicates = [t for t, c in seen.items() if c > 1]
        raise ValueError(f"Duplicate teams in seeding: {duplicates}.")

    empty = [t for t in teams if not t.strip()]
    if empty:
        raise ValueError(f"Seeding contains {len(empty)} empty or whitespace-only team name(s).")

    return teams