"""Model training.

Two model families, matching the spec's "Goal Model" and "Match Outcome
Model" layers:

- ``GoalModel``: Poisson regression predicting expected goals for each side.
- ``OutcomeModel``: a gradient-boosted multiclass classifier (LightGBM)
  predicting P(home win / draw / away win) directly, compared against the
  baselines in ``baselines.py``.

Both use the low-level LightGBM Booster API (not the sklearn wrapper) for
the outcome model specifically so the 3-way class space (home/draw/away) is
always fixed by ``num_class``, even if one outcome never appears in a small
training set -- the sklearn wrapper infers classes from the observed labels
only, which silently breaks on exactly that case.
"""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor

from worldcup_predictor.utils.config import GoalModelConfig, OutcomeModelConfig

OUTCOME_LABELS = ["away", "draw", "home"]
LABEL_TO_INT = {label: i for i, label in enumerate(OUTCOME_LABELS)}


@dataclass
class GoalModel:
    home_model: PoissonRegressor
    away_model: PoissonRegressor
    features: list[str]

    def predict(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        x = df[self.features].astype(float).to_numpy()
        return self.home_model.predict(x), self.away_model.predict(x)


def train_goal_model(train_df: pd.DataFrame, config: GoalModelConfig) -> GoalModel:
    """Fit independent Poisson regressions for home and away goals.

    ``train_df`` must contain only played matches (home_score/away_score not
    null) -- passing unplayed rows would either crash the fit or, worse,
    silently coerce NaN targets to something meaningless.
    """
    if train_df["home_score"].isna().any() or train_df["away_score"].isna().any():
        raise ValueError(
            "train_goal_model received rows with missing scores. Filter to "
            "played matches before training (see predict.py for the unplayed rows)."
        )
    x = train_df[config.features].astype(float).to_numpy()
    y_home = train_df["home_score"].astype(float).to_numpy()
    y_away = train_df["away_score"].astype(float).to_numpy()
    home_model = PoissonRegressor(alpha=config.regularization_alpha).fit(x, y_home)
    away_model = PoissonRegressor(alpha=config.regularization_alpha).fit(x, y_away)
    return GoalModel(home_model=home_model, away_model=away_model, features=list(config.features))


@dataclass
class OutcomeModel:
    booster: lgb.Booster
    features: list[str]

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        x = df[self.features].astype(float).to_numpy()
        proba = self.booster.predict(x)
        return pd.DataFrame(
            proba, columns=[f"p_{label}" for label in OUTCOME_LABELS], index=df.index
        )[["p_home", "p_draw", "p_away"]]


def train_outcome_model(train_df: pd.DataFrame, config: OutcomeModelConfig) -> OutcomeModel:
    if train_df["winner"].isna().any():
        raise ValueError(
            "train_outcome_model received rows with no result. Filter to played "
            "matches before training."
        )
    x = train_df[config.features].astype(float).to_numpy()
    y = train_df["winner"].map(LABEL_TO_INT).astype(int).to_numpy()

    params = dict(config.params)
    params.setdefault("objective", "multiclass")
    params.setdefault("num_class", len(OUTCOME_LABELS))
    n_estimators = params.pop("n_estimators", 100)
    params["verbosity"] = -1

    dataset = lgb.Dataset(x, label=y)
    booster = lgb.train(params, dataset, num_boost_round=n_estimators)
    return OutcomeModel(booster=booster, features=list(config.features))
