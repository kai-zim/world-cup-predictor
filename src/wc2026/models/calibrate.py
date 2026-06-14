"""EloPoissonModel calibration.

Fits the three free constants of EloPoissonModel (base_rate, supremacy_scale,
home_advantage_goals) against historical match data using maximum-likelihood
estimation on Poisson goal counts.

This should be run on the training window only (matches before the cut-off
date) to avoid leaking future results into the model constants.

Usage::

    features = assemble_match_features(elo_engine.replay(history))
    model = calibrate_elo_poisson(features)

TODO(calibrate): Log calibration metrics to MLflow (train NLL, calibration
curve) and persist fitted constants to configs/base.yaml so the dashboard
and CLI pick them up automatically.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from wc2026.models.goal_model import EloPoissonModel


def _poisson_nll(
    params: np.ndarray,
    home_elo: np.ndarray,
    away_elo: np.ndarray,
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    neutrals: np.ndarray,
) -> float:
    """Total negative Poisson log-likelihood for the three model parameters."""
    base_rate, supremacy_scale, home_goal_adv = params
    # Keep parameters in physically valid ranges.
    if base_rate <= 0.0 or supremacy_scale <= 0.0 or home_goal_adv < 0.0:
        return 1e9

    model = EloPoissonModel(
        base_rate=float(base_rate),
        supremacy_scale=float(supremacy_scale),
        home_advantage_goals=float(home_goal_adv),
    )
    nll = 0.0
    for he, ae, hg, ag, neutral in zip(home_elo, away_elo, home_goals, away_goals, neutrals):
        lam_h, lam_a = model.expected_goals(float(he), float(ae), bool(neutral))
        # Poisson log P(k; λ) = k·ln(λ) - λ - ln(k!)  (drop constant ln(k!)).
        nll -= float(hg) * np.log(max(lam_h, 1e-9)) - lam_h
        nll -= float(ag) * np.log(max(lam_a, 1e-9)) - lam_a
    return float(nll)


def calibrate_elo_poisson(
    features: pd.DataFrame,
    x0: tuple[float, float, float] = (1.35, 0.0024, 0.30),
) -> EloPoissonModel:
    """Fit EloPoissonModel constants via MLE on a feature frame.

    Args:
        features: Frame containing home_elo_pre, away_elo_pre, home_score,
            away_score, neutral. Must be training-window data only.
        x0: Initial parameter guess (base_rate, supremacy_scale, home_adv_goals).

    Returns:
        A calibrated EloPoissonModel with fitted constants.

    Raises:
        ValueError: If required columns are missing.
    """
    required = {"home_elo_pre", "away_elo_pre", "home_score", "away_score", "neutral"}
    missing = required - set(features.columns)
    if missing:
        raise ValueError(f"calibrate_elo_poisson: missing columns {sorted(missing)}")

    result = minimize(
        _poisson_nll,
        x0=np.array(x0),
        args=(
            features["home_elo_pre"].to_numpy(dtype=float),
            features["away_elo_pre"].to_numpy(dtype=float),
            features["home_score"].to_numpy(dtype=float),
            features["away_score"].to_numpy(dtype=float),
            features["neutral"].to_numpy(dtype=bool),
        ),
        method="Nelder-Mead",
        options={"maxiter": 10_000, "xatol": 1e-6, "fatol": 1e-6},
    )

    base_rate, supremacy_scale, home_goal_adv = result.x
    return EloPoissonModel(
        base_rate=float(max(base_rate, 0.1)),
        supremacy_scale=float(max(supremacy_scale, 1e-6)),
        home_advantage_goals=float(max(home_goal_adv, 0.0)),
    )