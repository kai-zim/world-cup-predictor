import pandas as pd
import pytest

from worldcup_predictor.models.baselines import (
    EloDiffBaseline,
    FifaRankingBaseline,
    HistoricalWinRateBaseline,
    PoissonBaseline,
    poisson_match_outcome_probs,
)
from worldcup_predictor.models.calibration import MulticlassCalibrator
from worldcup_predictor.models.evaluation import evaluate_predictions
from worldcup_predictor.models.predict import predict_goal_model, predict_outcome_model
from worldcup_predictor.models.train import train_goal_model, train_outcome_model


def _assert_valid_proba(df: pd.DataFrame) -> None:
    for col in ["p_home", "p_draw", "p_away"]:
        assert (df[col] >= 0).all()
        assert (df[col] <= 1).all()
    totals = df[["p_home", "p_draw", "p_away"]].sum(axis=1)
    assert totals.apply(lambda t: t == pytest.approx(1.0, abs=1e-6)).all()


def test_baselines_produce_valid_probabilities(played_matches):
    _assert_valid_proba(FifaRankingBaseline().predict_proba(played_matches))
    _assert_valid_proba(EloDiffBaseline().predict_proba(played_matches))
    _assert_valid_proba(HistoricalWinRateBaseline.fit(played_matches).predict_proba(played_matches))
    _assert_valid_proba(PoissonBaseline().predict_proba(played_matches))


def test_poisson_match_outcome_probs_sums_to_one():
    result = poisson_match_outcome_probs([1.5, 2.0], [1.0, 2.0])
    assert result[["p_home", "p_draw", "p_away"]].sum(axis=1).apply(lambda t: t == pytest.approx(1.0)).all()


def test_goal_model_trains_and_predicts(played_matches, feature_frame, model_config):
    goal_model = train_goal_model(played_matches, model_config.goal_model)
    predictions = predict_goal_model(goal_model, feature_frame)
    assert (predictions["expected_home_goals"] > 0).all()
    assert (predictions["expected_away_goals"] > 0).all()
    _assert_valid_proba(predictions)


def test_goal_model_rejects_unplayed_rows_in_training_data(feature_frame, model_config):
    with pytest.raises(ValueError):
        train_goal_model(feature_frame, model_config.goal_model)  # contains unplayed rows


def test_outcome_model_trains_and_predicts(played_matches, feature_frame, model_config):
    outcome_model = train_outcome_model(played_matches, model_config.outcome_model)
    predictions = predict_outcome_model(outcome_model, feature_frame)
    _assert_valid_proba(predictions)


def test_evaluate_predictions_metrics_are_sane(played_matches, model_config):
    outcome_model = train_outcome_model(played_matches, model_config.outcome_model)
    predictions = predict_outcome_model(outcome_model, played_matches)
    metrics = evaluate_predictions(played_matches["winner"], predictions)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["log_loss"] >= 0.0
    assert 0.0 <= metrics["brier_score"] <= 2.0
    assert 0.0 <= metrics["rps"] <= 1.0


def test_calibration_output_sums_to_one(played_matches, model_config):
    outcome_model = train_outcome_model(played_matches, model_config.outcome_model)
    predictions = predict_outcome_model(outcome_model, played_matches)
    calibrator = MulticlassCalibrator.fit(predictions, played_matches["winner"])
    calibrated = calibrator.transform(predictions)
    _assert_valid_proba(calibrated)
