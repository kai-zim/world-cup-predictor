import math


def shot_xg(distance_m: float, angle_rad: float, is_penalty: bool = False) -> float:
    if is_penalty:
        return 0.76
    distance_weight = -0.14 * distance_m
    angle_weight = 1.2 * angle_rad
    raw = -0.7 + distance_weight + angle_weight
    return 1.0 / (1.0 + math.exp(-raw))
