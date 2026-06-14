from dataclasses import dataclass


@dataclass(frozen=True)
class EloResult:
    home_rating: float
    away_rating: float


def expected_score(home_rating: float, away_rating: float) -> float:
    return 1.0 / (1.0 + 10 ** ((away_rating - home_rating) / 400.0))


def update_elo(home_rating: float, away_rating: float, home_result: float, k_factor: float = 20.0) -> EloResult:
    expected_home = expected_score(home_rating, away_rating)
    delta = k_factor * (home_result - expected_home)
    return EloResult(home_rating=home_rating + delta, away_rating=away_rating - delta)
