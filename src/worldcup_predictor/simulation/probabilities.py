"""Formatting helpers for simulation output (used by CLI/reporting/dashboard)."""

from __future__ import annotations

import pandas as pd

from worldcup_predictor.simulation.tournament import SimulationResult
from worldcup_predictor.utils.config import SimulationConfig


def probability_table(
    result: SimulationResult, config: SimulationConfig, teams: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Rounded, human-readable probability table, optionally enriched with
    team metadata (group, confederation) for reporting/dashboard display."""
    table = result.to_probability_table()
    prob_cols = [c for c in table.columns if c.startswith("prob_")]
    table[prob_cols] = table[prob_cols].round(config.reporting.probability_decimal_places)

    if teams is not None:
        table = table.merge(
            teams[["team_name", "confederation", "group_letter"]].rename(columns={"team_name": "team"}),
            on="team",
            how="left",
        )
    return table
