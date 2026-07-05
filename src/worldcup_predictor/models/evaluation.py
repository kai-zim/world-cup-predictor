"""Evaluation metrics and time-based train/test splitting.

No random train/test splits anywhere in this module: a World Cup prediction
model must be judged on tournaments it did not train on, so splitting is
always by ``season`` (year), never a random row shuffle.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss

from worldcup_predictor.models.baselines import (
    EloDiffBaseline,
    HistoricalWinRateBaseline,
    OutcomeBaseline,
    PoissonBaseline,
)
from worldcup_predictor.models.predict import predict_goal_model, predict_outcome_model
from worldcup_predictor.models.train import train_goal_model, train_outcome_model
from worldcup_predictor.utils.config import GoalModelConfig, ModelConfig, OutcomeModelConfig

OUTCOME_ORDER = ["home", "draw", "away"]  # matches OUTCOME_COLUMNS / p_home,p_draw,p_away
RPS_ORDER = ["away", "draw", "home"]  # ordinal: goal-difference direction, for the RPS metric


def time_based_split(
    df: pd.DataFrame, train_until: int, test_year: int, season_col: str = "season"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by season/year, never by row -- see module docstring."""
    train_df = df[df[season_col] <= train_until]
    test_df = df[df[season_col] == test_year]
    return train_df, test_df


def _predicted_label(proba: pd.DataFrame) -> pd.Series:
    columns = [f"p_{label}" for label in OUTCOME_ORDER]
    return proba[columns].idxmax(axis=1).str.removeprefix("p_")


def ranked_probability_score(y_true: pd.Series, proba: pd.DataFrame) -> float:
    """Mean RPS over all rows, using the ordinal away < draw < home ordering."""
    p = proba[[f"p_{label}" for label in RPS_ORDER]].to_numpy()
    e = np.array([[1.0 if label == true else 0.0 for label in RPS_ORDER] for true in y_true])
    cum_p = np.cumsum(p, axis=1)[:, :-1]
    cum_e = np.cumsum(e, axis=1)[:, :-1]
    per_row = ((cum_p - cum_e) ** 2).sum(axis=1) / (len(RPS_ORDER) - 1)
    return float(per_row.mean())


def brier_score(y_true: pd.Series, proba: pd.DataFrame) -> float:
    p = proba[[f"p_{label}" for label in OUTCOME_ORDER]].to_numpy()
    e = np.array([[1.0 if label == true else 0.0 for label in OUTCOME_ORDER] for true in y_true])
    return float(((p - e) ** 2).sum(axis=1).mean())


def evaluate_predictions(y_true: pd.Series, proba: pd.DataFrame) -> dict[str, float]:
    """accuracy, f1_macro, log_loss, brier_score, rps in one call."""
    predicted = _predicted_label(proba)
    # sklearn's log_loss sorts `labels` lexicographically internally and expects
    # y_prob's columns to already be in that order -- build the matrix in that
    # exact order (independent of OUTCOME_ORDER, which is used for brier/rps).
    lexicographic_order = sorted(OUTCOME_ORDER)
    proba_matrix = proba[[f"p_{label}" for label in lexicographic_order]].to_numpy()
    return {
        "accuracy": accuracy_score(y_true, predicted),
        "f1_macro": f1_score(y_true, predicted, average="macro", zero_division=0),
        "log_loss": log_loss(y_true, proba_matrix, labels=lexicographic_order),
        "brier_score": brier_score(y_true, proba),
        "rps": ranked_probability_score(y_true, proba),
    }


def run_time_based_backtest(
    historical_feature_frame: pd.DataFrame, model_config: ModelConfig
) -> pd.DataFrame:
    """Real historical backtest: for every split in
    ``model_config.evaluation.time_based_splits``, train on seasons up to
    ``train_until`` and evaluate on ``test_year`` -- never a random split.

    Only Elo/rolling-form/rest-days baselines and models are compared here
    (see ``configs/model.yaml: historical_backtest`` and
    ``features.feature_pipeline.build_historical_feature_frame``): FIFA
    ranking and squad market value are not available for most of 1930-2022,
    so the "ranking-only" and "market-value-only" baselines from the spec are
    not run against this dataset -- an honest scope cut given what the data
    actually contains, not a shortcut.

    Splits with no played matches in the train or test window (e.g. 2026,
    since it isn't finished yet) are skipped rather than silently producing
    empty/misleading metrics.
    """
    goal_features = model_config.historical_backtest.goal_model_features
    outcome_features = model_config.historical_backtest.outcome_model_features

    rows: list[dict] = []
    for split in model_config.evaluation.time_based_splits:
        train_df, test_df = time_based_split(historical_feature_frame, split.train_until, split.test_year)
        train_played = train_df[train_df["winner"].notna()]
        test_played = test_df[test_df["winner"].notna()]
        if train_played.empty or test_played.empty:
            continue

        split_label = f"train<={split.train_until}->test={split.test_year}"

        baselines: dict[str, OutcomeBaseline] = {
            "elo_diff_baseline": EloDiffBaseline(),
            "poisson_baseline": PoissonBaseline(),
            "historical_win_rate_baseline": HistoricalWinRateBaseline.fit(train_played),
        }
        for name, baseline in baselines.items():
            proba = baseline.predict_proba(test_played)
            metrics = evaluate_predictions(test_played["winner"], proba)
            rows.append({"split": split_label, "model": name, **metrics})

        goal_model = train_goal_model(
            train_played,
            GoalModelConfig(
                kind="poisson",
                features=goal_features,
                regularization_alpha=model_config.goal_model.regularization_alpha,
            ),
        )
        goal_proba = predict_goal_model(goal_model, test_played)
        goal_metrics = evaluate_predictions(test_played["winner"], goal_proba)
        rows.append({"split": split_label, "model": "goal_model", **goal_metrics})

        outcome_model_config = OutcomeModelConfig(
            kind="lightgbm", features=outcome_features, params=model_config.outcome_model.params
        )
        outcome_model = train_outcome_model(train_played, outcome_model_config)
        outcome_proba = predict_outcome_model(outcome_model, test_played)
        rows.append(
            {
                "split": split_label,
                "model": "outcome_model",
                **evaluate_predictions(test_played["winner"], outcome_proba),
            }
        )

    return pd.DataFrame(rows)
