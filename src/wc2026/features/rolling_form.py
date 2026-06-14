"""Rolling form feature helpers.

The canonical full-frame implementation lives in features.engineering
(build_form_features). This module adds convenience wrappers for:

* Extracting form for a single team at a specific point in time — useful
  for per-match prediction in the dashboard and CLI.
* Summarising the most recent form window as a human-readable dict.
"""

from __future__ import annotations

import pandas as pd

# Re-export so callers can import from one place.
from wc2026.features.engineering import build_form_features  # noqa: F401


def get_team_form(
    matches: pd.DataFrame,
    team: str,
    as_of: pd.Timestamp | str,
    window: int = 5,
) -> dict[str, float | None]:
    """Return rolling form metrics for ``team`` strictly before ``as_of``.

    Runs the full build_form_features pipeline and filters to the requested
    team and date. Returns None values where there is insufficient history
    (fewer than one previous match).

    Args:
        matches: Full chronologically-sorted match frame (core schema).
        team: Team name exactly as it appears in home_team / away_team columns.
        as_of: Cut-off timestamp — only matches *before* this date are used.
        window: Rolling window length (default 5 matches).

    Returns:
        Dict with keys ppg, winrate, gf (goals for), ga (goals against).
    """
    cutoff = pd.Timestamp(as_of)
    form = build_form_features(matches, windows=(window,))
    team_form = form[(form["team"] == team) & (form["date"] < cutoff)]

    if team_form.empty:
        return {"ppg": None, "winrate": None, "gf": None, "ga": None}

    latest = team_form.iloc[-1]
    return {
        "ppg": _safe_float(latest.get(f"ppg_{window}")),
        "winrate": _safe_float(latest.get(f"winrate_{window}")),
        "gf": _safe_float(latest.get(f"gf_{window}")),
        "ga": _safe_float(latest.get(f"ga_{window}")),
    }


def _safe_float(val: object) -> float | None:
    try:
        f = float(val)  # type: ignore[arg-type]
        return None if pd.isna(f) else f
    except (TypeError, ValueError):
        return None