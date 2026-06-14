import math


def poisson_probability(goals: int, lam: float) -> float:
    return (lam**goals) * math.exp(-lam) / math.factorial(goals)


def expected_points(home_lambda: float, away_lambda: float, max_goals: int = 10) -> tuple[float, float]:
    home_points = 0.0
    away_points = 0.0
    for home_goals in range(max_goals + 1):
        for away_goals in range(max_goals + 1):
            p = poisson_probability(home_goals, home_lambda) * poisson_probability(away_goals, away_lambda)
            if home_goals > away_goals:
                home_points += 3 * p
            elif home_goals < away_goals:
                away_points += 3 * p
            else:
                home_points += p
                away_points += p
    return home_points, away_points
