from dataclasses import dataclass
import random


@dataclass(frozen=True)
class Matchup:
    home_team: str
    away_team: str
    home_win_probability: float


def simulate_match(matchup: Matchup) -> str:
    return matchup.home_team if random.random() < matchup.home_win_probability else matchup.away_team


def simulate_knockout_tournament(matchups: list[Matchup], n_simulations: int = 1000) -> dict[str, float]:
    winners: dict[str, int] = {}
    for _ in range(n_simulations):
        final_winner = None
        for matchup in matchups:
            final_winner = simulate_match(matchup)
        if final_winner is None:
            continue
        winners[final_winner] = winners.get(final_winner, 0) + 1
    return {team: count / n_simulations for team, count in winners.items()}
