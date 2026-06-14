from world_cup_predictor.ratings.elo import expected_score, update_elo


def test_expected_score_balanced_when_equal_ratings() -> None:
    assert expected_score(1500, 1500) == 0.5


def test_update_elo_home_win_increases_home_rating() -> None:
    result = update_elo(home_rating=1500, away_rating=1500, home_result=1.0, k_factor=20)
    assert result.home_rating > 1500
    assert result.away_rating < 1500
