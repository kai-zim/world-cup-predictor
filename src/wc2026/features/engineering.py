"""Feature engineering.

Builds the per-match feature table consumed by *learned* goal/result models.
The analytic Elo->Poisson backend does not need these, but they are required
when ``ModelConfig.backend`` is a gradient-boosting model.

Leakage policy (hard rule)
--------------------------
Every feature for match *m* must be computable from information available
strictly *before* m kicks off. Rolling/form features therefore use a *shift(1)*
within each team's chronologically-sorted history so the current match never
contributes to its own features. The Elo columns are already pre-match
(produced by EloEngine.replay).

Fully implemented here: Elo diff, rolling form (win rate, goals for/against,
points-per-game over last 3/5/10), rest days, EWMA goals.

Stubbed (data not freely available — see TODO markers): xG diff, market-value
diff, FIFA-ranking diff, squad-quality diff, injury-impact diff.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from wc2026.utils.logging import get_logger

log = get_logger(__name__)


def _long_format(matches: pd.DataFrame) -> pd.DataFrame:
    """Explode each match into two team-perspective rows (home/away).

    Enables per-team rolling computations without duplicating logic.
    """
    home = matches.assign(
        team=matches["home_team"],
        opponent=matches["away_team"],
        goals_for=matches["home_score"],
        goals_against=matches["away_score"],
        is_home=True,
    )
    away = matches.assign(
        team=matches["away_team"],
        opponent=matches["home_team"],
        goals_for=matches["away_score"],
        goals_against=matches["home_score"],
        is_home=False,
    )
    cols = ["date", "team", "opponent", "goals_for", "goals_against", "is_home"]
    long = pd.concat([home[cols], away[cols]], ignore_index=True)
    return long.sort_values(["team", "date"]).reset_index(drop=True)


def _points(goals_for: pd.Series, goals_against: pd.Series) -> pd.Series:
    return np.select(
        [goals_for > goals_against, goals_for == goals_against],
        [3, 1],
        default=0,
    )


def build_form_features(matches: pd.DataFrame, windows: tuple[int, ...] = (3, 5, 10)) -> pd.DataFrame:
    """Compute leakage-safe rolling form features per team.

    Returns a long frame keyed by (date, team) with rolling stats *as known
    before* that date. Join back onto matches for both home and away teams.
    """
    long = _long_format(matches)
    long["points"] = _points(long["goals_for"], long["goals_against"])
    long["win"] = (long["goals_for"] > long["goals_against"]).astype(int)
    long["clean_sheet"] = (long["goals_against"] == 0).astype(int)

    g = long.groupby("team", group_keys=False)
    # rest days since previous match for this team
    long["rest_days"] = g["date"].diff().dt.days

    for w in windows:
        # shift(1) ensures the current row is excluded -> no leakage.
        def roll(col: str, fn: str = "mean") -> pd.Series:
            shifted = g[col].apply(lambda s: s.shift(1))
            return shifted.rolling(w, min_periods=1).agg(fn).reset_index(level=0, drop=True)

        long[f"ppg_{w}"] = roll("points")
        long[f"winrate_{w}"] = roll("win")
        long[f"gf_{w}"] = roll("goals_for")
        long[f"ga_{w}"] = roll("goals_against")
        long[f"cleansheets_{w}"] = roll("clean_sheet", "sum")

    # EWMA of goals for (half-life ~3 matches), pre-match.
    long["gf_ewma"] = g["goals_for"].apply(lambda s: s.shift(1).ewm(halflife=3).mean()).reset_index(level=0, drop=True)

    log.info("form_features_built", rows=len(long), windows=windows)
    return long


def assemble_match_features(matches_with_elo: pd.DataFrame) -> pd.DataFrame:
    """Produce the final model-ready feature matrix.

    Requires ``home_elo_pre``/``away_elo_pre`` columns (from EloEngine.replay).
    Joins rolling form for both sides and computes the ``*Diff`` features.
    """
    if "home_elo_pre" not in matches_with_elo:
        raise ValueError("run EloEngine.replay first to add pre-match Elo columns")

    form = build_form_features(matches_with_elo)
    keep = [c for c in form.columns if c not in {"opponent", "goals_for", "goals_against", "is_home", "points", "win", "clean_sheet"}]
    form = form[keep]

    df = matches_with_elo.copy()
    df = df.merge(form.add_prefix("home_"), left_on=["date", "home_team"], right_on=["home_date", "home_team"], how="left")
    df = df.merge(form.add_prefix("away_"), left_on=["date", "away_team"], right_on=["away_date", "away_team"], how="left")
    df = df.drop(columns=[c for c in df.columns if c.endswith("_date")], errors="ignore")

    # --- Diff features (home - away) ---
    df["EloDiff"] = df["home_elo_pre"] - df["away_elo_pre"]
    for w in (3, 5, 10):
        df[f"FormDiff_ppg_{w}"] = df[f"home_ppg_{w}"] - df[f"away_ppg_{w}"]
    df["RestDayDiff"] = df["home_rest_days"] - df["away_rest_days"]

    # --- Stubbed diff features (data not available) ---
    # TODO(features-xg): populate from ingested per-match xG. Until then null.
    df["xGDiff"] = np.nan
    # TODO(features-mv): requires market-value source (see loaders TODO).
    df["MarketValueDiff"] = np.nan
    # TODO(features-rank): requires FIFA ranking time-series join.
    df["RankingDiff"] = np.nan
    # TODO(features-squad): squad-quality aggregation (caps, top-5-league count).
    df["SquadQualityDiff"] = np.nan
    # TODO(features-injury): availability index from squad/injury reports.
    df["InjuryImpactDiff"] = np.nan

    log.info("match_features_assembled", rows=len(df), cols=df.shape[1])
    return df
