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
    if not matchups:
        return {}

    winners: dict[str, int] = {}
    for _ in range(n_simulations):
        current_round = [simulate_match(matchup) for matchup in matchups]
        while len(current_round) > 1:
            next_round: list[str] = []
            for i in range(0, len(current_round) - 1, 2):
                home = current_round[i]
                away = current_round[i + 1]
                probability = 0.5
                next_round.append(home if random.random() < probability else away)
            if len(current_round) % 2 == 1:
                next_round.append(current_round[-1])
            current_round = next_round
        winners[current_round[0]] = winners.get(current_round[0], 0) + 1

    return {team: count / n_simulations for team, count in winners.items()}
