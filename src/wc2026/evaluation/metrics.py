"""Evaluation metrics.

Three families, matching the project spec:

* Classification (W/D/L): accuracy, log-loss, Brier (multiclass), plus a
  reliability-curve helper.
* Regression (goals/xG): MAE, RMSE, mean Poisson deviance.
* Simulation: comparison of forecast champion probabilities against the
  realised outcome and (optionally) bookmaker-implied probabilities.

All functions are pure and operate on numpy arrays / pandas frames so they are
trivially unit-testable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    mean_absolute_error,
    mean_poisson_deviance,
)


def classification_metrics(
    y_true: np.ndarray, proba: np.ndarray, labels: list[int]
) -> dict[str, float]:
    """Accuracy, log-loss and multiclass Brier score.

    ``proba`` has shape (n, n_classes); columns aligned to ``labels``.
    """
    y_pred = np.asarray(labels)[proba.argmax(axis=1)]
    onehot = np.zeros_like(proba)
    label_index = {lab: i for i, lab in enumerate(labels)}
    for i, y in enumerate(y_true):
        onehot[i, label_index[int(y)]] = 1.0
    brier = float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "log_loss": float(log_loss(y_true, proba, labels=labels)),
        "brier": brier,
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """MAE, RMSE, mean Poisson deviance (clipped to positive predictions)."""
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    y_pred_pos = np.clip(y_pred, 1e-6, None)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse,
        "poisson_deviance": float(mean_poisson_deviance(y_true + 1e-9, y_pred_pos)),
    }


def reliability_curve(
    y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10
) -> pd.DataFrame:
    """Binned calibration: mean predicted vs empirical frequency.

    For binary/one-vs-rest probabilities. Returns a frame with one row per bin.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(proba, bins) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        rows.append(
            {
                "bin": b,
                "mean_predicted": float(proba[mask].mean()),
                "empirical_freq": float(y_true[mask].mean()),
                "count": int(mask.sum()),
            }
        )
    return pd.DataFrame(rows)


def champion_log_loss(forecast: pd.DataFrame, actual_champion: str) -> float:
    """Log-loss of the champion forecast against the realised winner.

    ``forecast`` must contain columns ``team`` and ``p_champion``. A lower value
    means the model assigned higher probability to the eventual champion.
    """
    p = forecast.loc[forecast["team"] == actual_champion, "p_champion"]
    prob = float(p.iloc[0]) if len(p) else 1e-9
    return float(-np.log(max(prob, 1e-9)))
