"""Probability calibration layer (isotonic regression, one-vs-rest).

Gradient-boosted classifiers are notoriously overconfident; this recalibrates
each outcome's probability against observed frequency and renormalizes the
three probabilities back to sum to 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.isotonic import IsotonicRegression

OUTCOME_COLUMNS = ["p_home", "p_draw", "p_away"]
_COLUMN_TO_LABEL = {"p_home": "home", "p_draw": "draw", "p_away": "away"}


@dataclass
class MulticlassCalibrator:
    calibrators: dict[str, IsotonicRegression]

    @classmethod
    def fit(cls, proba: pd.DataFrame, y_true: pd.Series) -> MulticlassCalibrator:
        calibrators = {}
        for col in OUTCOME_COLUMNS:
            target = (y_true == _COLUMN_TO_LABEL[col]).astype(float)
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(proba[col].astype(float), target)
            calibrators[col] = iso
        return cls(calibrators=calibrators)

    def transform(self, proba: pd.DataFrame) -> pd.DataFrame:
        calibrated = pd.DataFrame(
            {col: self.calibrators[col].predict(proba[col].astype(float)) for col in OUTCOME_COLUMNS},
            index=proba.index,
        )
        total = calibrated.sum(axis=1)
        # A row of all-zero calibrated scores is a degenerate edge case (e.g.
        # extremely small calibration sets); fall back to a uniform distribution
        # rather than dividing by zero.
        degenerate = total <= 0
        calibrated = calibrated.div(total.where(~degenerate, 1.0), axis=0)
        calibrated.loc[degenerate, OUTCOME_COLUMNS] = 1.0 / len(OUTCOME_COLUMNS)
        return calibrated
