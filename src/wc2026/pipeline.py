"""Pipeline orchestration.

Ties the modules together into the two operations the spec calls for:

1. ``run_forecast`` — given a match history window and a 32-team seeding, fit
   Elo, build the goal model, run Monte-Carlo, and return the forecast table.

2. ``update_after_round`` — the *dynamic* update: append newly-played knockout
   results, refresh Elo (and form), then re-simulate only the remaining
   bracket. This is what produces fresh forecasts after the group stage, R32,
   R16, QF and SF.
"""

from __future__ import annotations

import pandas as pd

from wc2026.config.schema import AppConfig
from wc2026.models.elo import EloEngine
from wc2026.models.goal_model import EloPoissonModel, GoalModel
from wc2026.simulation.bracket import Bracket, assign_teams, build_empty_r32_bracket
from wc2026.simulation.simulator import TournamentSimulator
from wc2026.utils.logging import get_logger

log = get_logger(__name__)


def _build_goal_model(cfg: AppConfig) -> GoalModel:
    """Select the goal-model backend. Defaults to analytic Elo->Poisson."""
    if cfg.model.backend == "elo_poisson":
        return EloPoissonModel()
    # TODO(model-backend): load a trained LearnedGoalModel from models/ when
    # backend is lightgbm/xgboost. Falls back to analytic model for now.
    log.warning("learned_backend_not_loaded", backend=cfg.model.backend)
    return EloPoissonModel()


def run_forecast(
    history: pd.DataFrame,
    seeding: list[str],
    cfg: AppConfig,
    bracket: Bracket | None = None,
) -> tuple[pd.DataFrame, dict[str, float], Bracket]:
    """Fit Elo on history and simulate the bracket.

    Returns (forecast_table, final_elo_ratings, bracket).
    """
    engine = EloEngine(cfg.elo)
    engine.replay(history)
    ratings = engine.ratings_snapshot()

    if bracket is None:
        bracket = build_empty_r32_bracket()
        assign_teams(bracket, seeding)

    model = _build_goal_model(cfg)
    sim = TournamentSimulator(model, ratings, cfg.simulation, cfg.poisson)
    forecast = sim.run(bracket)
    return forecast, ratings, bracket


def update_after_round(
    history: pd.DataFrame,
    new_results: pd.DataFrame,
    remaining_bracket: Bracket,
    cfg: AppConfig,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Incorporate a completed round and re-simulate the rest.

    ``new_results`` are the just-played knockout matches in the same schema as
    ``history``. ``remaining_bracket`` must already have its newly-known leaves
    assigned to the winners (caller resolves which slot each winner occupies).
    """
    combined = (
        pd.concat([history, new_results], ignore_index=True)
        .sort_values("date")
        .reset_index(drop=True)
    )
    engine = EloEngine(cfg.elo)
    engine.replay(combined)
    ratings = engine.ratings_snapshot()

    model = _build_goal_model(cfg)
    sim = TournamentSimulator(model, ratings, cfg.simulation, cfg.poisson)
    forecast = sim.run(remaining_bracket)
    log.info("update_after_round_done", new_matches=len(new_results))
    return forecast, ratings
