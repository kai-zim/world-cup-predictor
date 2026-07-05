"""Monte Carlo tournament simulator.

Walks the real bracket stage by stage (round_of_32 -> ... -> final):

- Already-completed knockout matches are fixed (their real winner counts in
  every single simulation draw) -- see spec requirement "bekannte Ergebnisse
  fixieren".
- Scheduled-but-not-yet-played matches that already exist in the data (the
  next round, once it has been drawn) are simulated with the real two teams.
- Stages that don't exist in the data yet (more than one round ahead of the
  last completed round) are synthesized by pairing the previous stage's
  simulated winners in bracket order (adjacent match_id pairs) -- a
  documented approximation of the official bracket adjacency, since the raw
  dataset does not expose an explicit next-match/slot linkage to verify
  against (see README limitations).

Team strength (Elo, ranking, market value, rolling form) is frozen as a
snapshot of "now" for the whole simulation: ratings are not re-estimated
match-by-match *within* a single hypothetical future run. That is the
standard simplification for forward tournament simulation (compute cost vs.
realism) and is revisited in the roadmap, not silently assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from worldcup_predictor.models.train import GoalModel
from worldcup_predictor.simulation.knockout import resolve_knockout
from worldcup_predictor.utils.config import SimulationConfig
from worldcup_predictor.utils.logging import get_logger

logger = get_logger(__name__)

STAGE_PROGRESSION = ["round_of_32", "round_of_16", "quarter_final", "semi_final", "final"]

_SNAPSHOT_HOME_COLS = {
    "elo": "home_elo_post",
    "ranking": "home_fifa_ranking",
    "market_value": "home_squad_total_market_value_eur",
    "form5_win_rate": "home_form5_win_rate",
    "form5_goal_diff_per_game": "home_form5_goal_diff_per_game",
    "form5_xg_per_game": "home_form5_xg_per_game",
}
_SNAPSHOT_AWAY_COLS = {
    "elo": "away_elo_post",
    "ranking": "away_fifa_ranking",
    "market_value": "away_squad_total_market_value_eur",
    "form5_win_rate": "away_form5_win_rate",
    "form5_goal_diff_per_game": "away_form5_goal_diff_per_game",
    "form5_xg_per_game": "away_form5_xg_per_game",
}


def build_team_snapshots(feature_frame: pd.DataFrame) -> pd.DataFrame:
    """Each team's most recent known state (from its latest listed match,
    played or scheduled), used as the frozen strength for simulation."""
    long_rows = []
    ordered = feature_frame.sort_values("date")
    for _, row in ordered.iterrows():
        home_state = {k: row.get(v) for k, v in _SNAPSHOT_HOME_COLS.items()}
        away_state = {k: row.get(v) for k, v in _SNAPSHOT_AWAY_COLS.items()}
        long_rows.append({"team": row["home_team"], "date": row["date"], **home_state})
        long_rows.append({"team": row["away_team"], "date": row["date"], **away_state})
    long_df = pd.DataFrame(long_rows).sort_values("date")
    return long_df.groupby("team").last()


def _safe_array(values: pd.Series) -> np.ndarray:
    return values.astype(float).fillna(0.0).to_numpy()


def _compute_lambdas(
    home_teams: np.ndarray, away_teams: np.ndarray, snapshots: pd.DataFrame, goal_model: GoalModel
) -> tuple[np.ndarray, np.ndarray]:
    home = snapshots.loc[home_teams]
    away = snapshots.loc[away_teams]
    n = len(home_teams)
    features = pd.DataFrame(
        {
            "elo_diff": _safe_array(home["elo"]) - _safe_array(away["elo"]),
            "ranking_diff": _safe_array(away["ranking"]) - _safe_array(home["ranking"]),
            "market_value_diff_log": np.log(
                (_safe_array(home["market_value"]) + 1.0) / (_safe_array(away["market_value"]) + 1.0)
            ),
            # future kickoff times/rest are not yet known -- documented assumption
            "rest_day_diff": np.zeros(n),
            "is_neutral": np.ones(n),
            "form5_win_rate_diff": _safe_array(home["form5_win_rate"]) - _safe_array(away["form5_win_rate"]),
            "form5_goal_diff_per_game_diff": (
                _safe_array(home["form5_goal_diff_per_game"]) - _safe_array(away["form5_goal_diff_per_game"])
            ),
            "form5_xg_per_game_diff": (
                _safe_array(home["form5_xg_per_game"]) - _safe_array(away["form5_xg_per_game"])
            ),
        }
    )
    return goal_model.predict(features)


@dataclass
class SimulationResult:
    n_simulations: int
    stage_reach_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    champion_counts: dict[str, int] = field(default_factory=dict)

    def to_probability_table(self) -> pd.DataFrame:
        stages = [*STAGE_PROGRESSION, "champion"]
        rows = []
        for team, counts in self.stage_reach_counts.items():
            row: dict[str, str | float] = {"team": team}
            for stage in stages:
                numerator = self.champion_counts.get(team, 0) if stage == "champion" else counts.get(stage, 0)
                row[f"prob_{stage}"] = numerator / self.n_simulations
            rows.append(row)
        df = pd.DataFrame(rows)
        sort_col = "prob_champion" if "prob_champion" in df.columns else f"prob_{stages[0]}"
        return df.sort_values(sort_col, ascending=False).reset_index(drop=True)


def simulate_tournament(
    matches: pd.DataFrame, feature_frame: pd.DataFrame, goal_model: GoalModel, config: SimulationConfig
) -> SimulationResult:
    n = config.monte_carlo.n_simulations
    rng = np.random.default_rng(config.monte_carlo.random_seed)
    snapshots = build_team_snapshots(feature_frame)

    knockout = matches[matches["is_knockout"] & (matches["stage"] != "third_place")]
    reach_counts: dict[str, dict[str, int]] = {team: {} for team in snapshots.index}
    champion_counts: dict[str, int] = dict.fromkeys(snapshots.index, 0)

    previous_winners: np.ndarray | None = None  # shape (n_matches_prev_stage, n)

    for stage in STAGE_PROGRESSION:
        stage_matches = knockout[knockout["stage"] == stage].sort_values("match_id")

        if not stage_matches.empty:
            home = np.tile(stage_matches["home_team"].to_numpy()[:, None], (1, n))
            away = np.tile(stage_matches["away_team"].to_numpy()[:, None], (1, n))
            fixed_winner = stage_matches["winner"].to_numpy()  # "home"/"away"/None
        elif previous_winners is not None and len(previous_winners) >= 2:
            n_next = len(previous_winners) // 2
            home = previous_winners[0 : 2 * n_next : 2]
            away = previous_winners[1 : 2 * n_next : 2]
            fixed_winner = np.array([None] * n_next)
        else:
            # No data for this stage yet and nothing simulated to build it from
            # (e.g. we're still mid-group-stage, or this stage is skipped
            # entirely in the fixture) -- move on, a later stage may still
            # have real data (as in the test fixture, which jumps straight
            # from the group stage to the semi-finals).
            continue

        n_matches = home.shape[0]
        winners = np.empty_like(home)

        for i in range(n_matches):
            if pd.notna(fixed_winner[i]):
                winner_team = home[i, 0] if fixed_winner[i] == "home" else away[i, 0]
                winners[i, :] = winner_team
                continue

            home_teams_i, away_teams_i = home[i], away[i]
            same_home = np.all(home_teams_i == home_teams_i[0])
            same_away = np.all(away_teams_i == away_teams_i[0])
            lambda_home: float | np.ndarray
            lambda_away: float | np.ndarray
            if same_home and same_away:
                lh, la = _compute_lambdas(home_teams_i[:1], away_teams_i[:1], snapshots, goal_model)
                lambda_home, lambda_away = float(lh[0]), float(la[0])
            else:
                lambda_home, lambda_away = _compute_lambdas(home_teams_i, away_teams_i, snapshots, goal_model)

            outcome = resolve_knockout(lambda_home, lambda_away, n, rng, config.knockout)
            winners[i, :] = np.where(outcome.winner_is_home, home_teams_i, away_teams_i)

        # Every team appearing in this stage "reached" it (survived the previous one).
        # Vectorized reach counting: for each unique team, count draws (columns) in
        # which it appears anywhere among this stage's home/away slots.
        all_participants = np.concatenate([home, away], axis=0)  # (2*n_matches, n)
        for team in np.unique(all_participants):
            reach_counts.setdefault(team, {})
            reach_counts[team][stage] = int((all_participants == team).any(axis=0).sum())

        if stage == "final":
            for team in np.unique(winners):
                champion_counts[team] = int((winners == team).sum())

        previous_winners = winners

    return SimulationResult(n_simulations=n, stage_reach_counts=reach_counts, champion_counts=champion_counts)
