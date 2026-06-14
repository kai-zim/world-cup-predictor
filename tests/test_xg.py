from world_cup_predictor.metrics.xg import shot_xg


def test_penalty_xg_is_constant() -> None:
    assert shot_xg(distance_m=11, angle_rad=1.0, is_penalty=True) == 0.76


def test_xg_increases_with_better_shot_angle() -> None:
    narrow = shot_xg(distance_m=12, angle_rad=0.2)
    wide = shot_xg(distance_m=12, angle_rad=0.8)
    assert wide > narrow


def test_xg_decreases_with_distance() -> None:
    close = shot_xg(distance_m=6, angle_rad=0.7)
    far = shot_xg(distance_m=20, angle_rad=0.7)
    assert close > far
