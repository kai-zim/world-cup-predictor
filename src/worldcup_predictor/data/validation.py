"""Explicit missing-value and leakage validation utilities.

Per the project's data rules: missing values must be surfaced, never silently
dropped or imputed without a trace, and no feature computation may see data
from the future relative to the match it describes.
"""

from __future__ import annotations

import pandas as pd

from worldcup_predictor.utils.logging import get_logger

logger = get_logger(__name__)


def missing_value_report(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Return a per-column missing-value report and log a warning for
    columns that are more than 50% empty (loud, not silent)."""
    total = len(df)
    counts = df.isna().sum()
    report = pd.DataFrame(
        {
            "column": counts.index,
            "missing_count": counts.values,
            "missing_pct": (counts.values / total * 100) if total else 0.0,
        }
    ).sort_values("missing_pct", ascending=False, ignore_index=True)

    heavy = report[report["missing_pct"] > 50]
    for _, row in heavy.iterrows():
        logger.warning(
            "%s.%s is %.1f%% missing (%d/%d rows)",
            name,
            row["column"],
            row["missing_pct"],
            row["missing_count"],
            total,
        )
    return report


def assert_chronological(df: pd.DataFrame, date_col: str = "date") -> None:
    """Raise if the frame is not sorted ascending by date.

    Elo updates and rolling-form features are only correct if computed in
    strict chronological order.
    """
    dates = pd.to_datetime(df[date_col])
    if not dates.is_monotonic_increasing:
        first_violation = (dates.diff() < pd.Timedelta(0)).idxmax()
        raise ValueError(
            f"{date_col} is not sorted ascending (first violation at index "
            f"{first_violation}). Sort the frame before computing chronological "
            "features -- otherwise Elo/rolling-form would leak future results."
        )


def assert_no_future_leakage(
    feature_frame: pd.DataFrame, as_of_col: str, source_date_col: str
) -> None:
    """Raise if any row's source data postdates the match it's attached to.

    ``as_of_col`` is the date of the match the feature is being computed for;
    ``source_date_col`` is the date of the historical match the feature value
    was derived from. Every value in source_date_col must be strictly earlier.
    """
    violations = feature_frame[feature_frame[source_date_col] >= feature_frame[as_of_col]]
    if not violations.empty:
        raise ValueError(
            f"Data leakage detected: {len(violations)} rows use {source_date_col} "
            f">= {as_of_col}. Example offending rows:\n{violations.head()}"
        )
