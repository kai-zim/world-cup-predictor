import pandas as pd

from world_cup_predictor.models.training import TrainedModel


def predict_matchup_probabilities(model: TrainedModel, features: pd.DataFrame) -> pd.DataFrame:
    proba = model.pipeline.predict_proba(features[model.feature_columns])
    return pd.DataFrame(
        {
            "home_win_prob": proba[:, 1],
            "away_or_draw_prob": proba[:, 0],
        }
    )
