from world_cup_predictor.simulation.poisson import expected_points


def test_expected_points_are_non_negative() -> None:
    home_pts, away_pts = expected_points(home_lambda=1.5, away_lambda=1.1)
    assert home_pts >= 0
    assert away_pts >= 0


def test_expected_points_total_is_bounded() -> None:
    home_pts, away_pts = expected_points(home_lambda=0.0, away_lambda=0.0)
    assert home_pts + away_pts <= 2.01
