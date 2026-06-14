"""Learned goal model (LightGBM/XGBoost) — optional backend.

This wraps a trained regressor that predicts expected goals from the engineered
feature vector, exposing the same :class:`~wc2026.models.goal_model.GoalModel`
interface as the analytic Elo->Poisson model so the simulator is agnostic.

Status: scaffolded. Training requires the feature table from
``features.engineering.assemble_match_features``. Because several headline
features (xG, market value) are currently null placeholders, a model trained
today effectively uses Elo + form only — which is fine and often competitive,
but the class is structured to absorb the richer features once available.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class LearnedGoalModel:
    """Two regressors (home goals, away goals) behind the GoalModel interface.

    The constructor takes already-fitted estimators; use :meth:`train` to fit.
    For the simulator's ``expected_goals(home_elo, away_elo, neutral)`` call we
    build a minimal feature row. A production version would pass the full
    pre-computed feature vector per fixture rather than re-deriving from Elo.
    """

    feature_columns: list[str]
    home_model: object | None = None  # fitted regressor
    away_model: object | None = None

    @classmethod
    def train(
        cls,
        features: pd.DataFrame,
        feature_columns: list[str],
        backend: str = "lightgbm",
        params: dict | None = None,
    ) -> LearnedGoalModel:
        """Fit home/away expected-goals regressors.

        TODO(model-train): wire MLflow autologging + early stopping with a
        time-based validation split (never random — temporal leakage). The
        body below shows the intended shape; flesh out per chosen backend.
        """
        params = params or {}
        x = features[feature_columns].to_numpy()
        y_home = features["home_score"].to_numpy()
        y_away = features["away_score"].to_numpy()

        if backend == "lightgbm":
            from lightgbm import LGBMRegressor

            home = LGBMRegressor(objective="poisson", **params).fit(x, y_home)
            away = LGBMRegressor(objective="poisson", **params).fit(x, y_away)
        elif backend == "xgboost":
            from xgboost import XGBRegressor

            home = XGBRegressor(objective="count:poisson", **params).fit(x, y_home)
            away = XGBRegressor(objective="count:poisson", **params).fit(x, y_away)
        else:
            raise ValueError(f"unsupported backend: {backend}")

        return cls(feature_columns=feature_columns, home_model=home, away_model=away)

    def expected_goals(
        self, home_elo: float, away_elo: float, neutral: bool
    ) -> tuple[float, float]:
        """Predict (lambda_home, lambda_away).

        NOTE: the simulator currently calls with only Elo + neutrality. To use
        the full feature vector, extend the simulator to pass a feature row and
        replace this minimal builder. As a safe fallback when models are unset,
        return a neutral baseline so the pipeline never crashes.
        """
        if self.home_model is None or self.away_model is None:
            return 1.35, 1.35
        # Minimal feature row: only EloDiff is well-defined from these inputs.
        row = {c: np.nan for c in self.feature_columns}
        if "EloDiff" in row:
            row["EloDiff"] = home_elo - away_elo
        x = pd.DataFrame([row])[self.feature_columns].to_numpy()
        lam_h = float(self.home_model.predict(x)[0])  # type: ignore[attr-defined]
        lam_a = float(self.away_model.predict(x)[0])  # type: ignore[attr-defined]
        return max(0.05, lam_h), max(0.05, lam_a)

    def save(self, path: str | Path) -> None:
        """Persist via joblib. TODO(model-versioning): register in MLflow."""
        import joblib

        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> LearnedGoalModel:
        import joblib

        return joblib.load(path)  # type: ignore[no-any-return]
