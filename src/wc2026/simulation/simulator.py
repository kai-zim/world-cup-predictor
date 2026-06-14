"""Monte-Carlo tournament simulator.

Given a bracket with assigned teams, a goal model, and current Elo ratings,
this runs N independent single-elimination simulations and aggregates, for each
team, the probability of reaching each round and winning the title.

Design notes
------------
* The bracket tree is traversed bottom-up per simulation. Each internal node
  resolves a knockout fixture between the winners of its two children.
* Ratings are treated as static during a single tournament simulation. A more
  advanced variant would update Elo between rounds within a sim; that is a
  documented extension (TODO(sim-dynamic)), separate from the *between-real-
  rounds* re-fitting handled by the update pipeline.
* Results are reproducible via the seeded ``numpy`` Generator.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from wc2026.config.schema import PoissonConfig, SimulationConfig
from wc2026.models.goal_model import GoalModel, resolve_knockout
from wc2026.simulation.bracket import Bracket, BracketSlot
from wc2026.utils.logging import get_logger

log = get_logger(__name__)

# Round index -> human label, aligned with Bracket.round_labels but counted
# from the FINAL upward so we can tag "reached round X".
_REACHED_ROUNDS = [
    "round_of_16",
    "quarter_final",
    "semi_final",
    "final",
    "champion",
]


class TournamentSimulator:
    """Runs Monte-Carlo simulations over a knockout bracket."""

    def __init__(
        self,
        goal_model: GoalModel,
        elo_ratings: dict[str, float],
        sim_cfg: SimulationConfig,
        poisson_cfg: PoissonConfig,
    ) -> None:
        self._model = goal_model
        self._elo = elo_ratings
        self._sim_cfg = sim_cfg
        self._poisson_cfg = poisson_cfg
        self._rng = np.random.default_rng(sim_cfg.random_seed)

    def _elo_of(self, team: str) -> float:
        return self._elo.get(team, 1500.0)

    def _play(self, home: str, away: str) -> str:
        """Resolve a single knockout fixture; return the winning team."""
        eh, ea = self._elo_of(home), self._elo_of(away)
        lam_h, lam_a = self._model.expected_goals(eh, ea, neutral=True)
        res = resolve_knockout(
            home, away, lam_h, lam_a, self._poisson_cfg, self._rng, eh, ea
        )
        return res.winner

    def _simulate_once(
        self, root: BracketSlot, reached: dict[str, set[str]]
    ) -> str:
        """Play one full tournament; record rounds reached. Return champion.

        ``reached[team]`` accumulates the labels of rounds the team appeared in
        as a *participant of that round's match*.

        Bracket depths (root = champion node):
            depth 0 -> champion node (resolves the final)
            depth 1 -> final-match nodes
            depth 2 -> semi-final nodes
            depth 3 -> quarter-final nodes
            depth 4 -> round-of-16 nodes
            depth 5 -> leaves (round-of-32 teams)
        """

        # Map a node's depth to the round its match represents.
        depth_to_label = {
            0: "final",  # champion node resolves the final fixture
            1: "semi_final",
            2: "quarter_final",
            3: "round_of_16",
        }

        def resolve(node: BracketSlot, depth: int) -> str:
            if node.is_leaf:
                assert node.team is not None, "bracket not fully assigned"
                return node.team
            assert node.left and node.right
            home = resolve(node.left, depth + 1)
            away = resolve(node.right, depth + 1)
            label = depth_to_label.get(depth, "round_of_16")
            reached[home].add(label)
            reached[away].add(label)
            return self._play(home, away)

        champion = resolve(root, 0)
        reached[champion].add("champion")
        return champion

    def run(self, bracket: Bracket) -> pd.DataFrame:
        """Run N simulations; return a per-team probability table."""
        n = self._sim_cfg.n_simulations
        counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

        for _ in range(n):
            reached: dict[str, set[str]] = defaultdict(set)
            self._simulate_once(bracket.root, reached)
            for team, rounds in reached.items():
                for r in rounds:
                    counts[team][r] += 1

        records = []
        for team, rounds in counts.items():
            rec: dict[str, float | str] = {"team": team}
            for r in _REACHED_ROUNDS:
                rec[f"p_{r}"] = rounds.get(r, 0) / n
            records.append(rec)

        df = pd.DataFrame(records).sort_values("p_champion", ascending=False)
        df = df.reset_index(drop=True)
        log.info("simulation_done", n=n, teams=len(df))
        return df
