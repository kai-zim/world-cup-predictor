"""Training pipeline entry point.

Ties together data loading, validation, Elo replay, feature engineering, and
LearnedGoalModel fitting. The training window is always defined by date to
prevent leakage — never by a random split.

Typical usage::

    model = train(results_path="data/processed/results.csv", cfg=AppConfig())
    model.save("models/goal_model.joblib")

TODO(model-train): Add MLflow autologging, temporal CV fold evaluation, and
early-stopping for LightGBM/XGBoost. Persist run ID and metrics alongside
the serialised model.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from wc2026.config.schema import AppConfig
from wc2026.data.loaders import filter_window, load_results
from wc2026.data.validate_data import validate_match_frame
from wc2026.features.engineering import assemble_match_features
from wc2026.models.elo import EloEngine
from wc2026.models.learned import LearnedGoalModel
from wc2026.utils.logging import get_logger

log = get_logger(__name__)

# Columns passed to the learned model. Must match what assemble_match_features
# produces. Extend this list as richer features become available.
DEFAULT_FEATURE_COLUMNS: list[str] = [
    "EloDiff",
    "FormDiff_ppg_3",
    "FormDiff_ppg_5",
    "FormDiff_ppg_10",
    "RestDayDiff",
]


def train(
    results_path: Path | str,
    cfg: AppConfig,
    train_start: str = "2022-12-19",
    train_end: str | None = None,
    feature_columns: list[str] | None = None,
) -> LearnedGoalModel:
    """Full training run: load → validate → Elo replay → features → fit.

    Args:
        results_path: Path to the normalised results CSV (core schema).
        cfg: Application config. cfg.model.backend selects lightgbm or xgboost.
        train_start: ISO date string for the start of the training window.
            Default is the day after WC 2022 final — exclude the tournament itself.
        train_end: ISO date string for the end of the training window.
            Defaults to today. Set explicitly to the end of the group stage to
            avoid including knockout results in training data.
        feature_columns: Override the default feature column list.

    Returns:
        A fitted LearnedGoalModel ready for serialisation.
    """
    cols = feature_columns or DEFAULT_FEATURE_COLUMNS

    df = load_results(results_path)
    validate_match_frame(df)

    end = train_end or str(pd.Timestamp.today().normalize().date())
    df = filter_window(df, start=train_start, end=end)

    engine = EloEngine(cfg.elo)
    df_elo = engine.replay(df)
    features = assemble_match_features(df_elo)

    # Fall back to lightgbm if the analytic backend is selected — the learned
    # model requires a gradient-boosting backend.
    backend = cfg.model.backend if cfg.model.backend != "elo_poisson" else "lightgbm"

    model = LearnedGoalModel.train(
        features=features,
        feature_columns=cols,
        backend=backend,
        params=dict(cfg.model.params),
    )
    log.info("model_trained", backend=backend, rows=len(features), features=cols)
    return model