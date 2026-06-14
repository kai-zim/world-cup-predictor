from world_cup_predictor.simulation.poisson import expected_points


def test_expected_points_are_non_negative() -> None:
    home_pts, away_pts = expected_points(home_lambda=1.5, away_lambda=1.1)
    assert home_pts >= 0
    assert away_pts >= 0
