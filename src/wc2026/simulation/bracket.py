"""World Cup 2026 knockout bracket definition.

WC2026 introduces a 48-team format: 12 groups of 4. The 12 group winners, 12
runners-up, plus the 8 best third-placed teams advance to a 32-team knockout
(Round of 32 -> R16 -> QF -> SF -> Final).

This module provides two assignment paths:

1. **Slot-based (production)** — :func:`assign_teams_from_slot_map` maps
   official slot names (e.g. ``"1A"``, ``"2B"``, ``"3G"``) to team names and
   places them onto the correct bracket leaves according to
   :data:`WC2026_R32_SLOT_ORDER`. This is the correct path for a real forecast.

2. **Sequential (legacy / demo)** — :func:`assign_teams` places 32 teams in
   the order given without regard to the official pairing. Still used by tests
   and the synthetic demo run.

Slot naming convention:
    ``"1X"``   — winner of group X
    ``"2X"``   — runner-up of group X
    ``"3X"``   — the qualifying third-placed team from group X

TODO(bracket-official): verify :data:`WC2026_R32_SLOT_ORDER` against the
official FIFA 2026 match schedule at
https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026
before a production run. The leaf order determines every simulated R32 pairing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias

# ---------------------------------------------------------------------------
# Bracket tree
# ---------------------------------------------------------------------------


@dataclass
class BracketSlot:
    """A leaf (qualified team) or internal node (winner of a sub-tie)."""

    name: str
    team: str | None = None  # filled at leaves once groups resolve
    left: BracketSlot | None = None
    right: BracketSlot | None = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


@dataclass
class Bracket:
    """A full single-elimination bracket with named rounds."""

    root: BracketSlot
    round_labels: list[str] = field(
        default_factory=lambda: [
            "round_of_32",
            "round_of_16",
            "quarter_final",
            "semi_final",
            "final",
        ]
    )


def build_empty_r32_bracket() -> Bracket:
    """Construct an empty 32-team single-elimination bracket.

    Leaves are named ``seed1..seed32`` in bracket order (adjacent pairs meet
    in the Round of 32). Team assignment happens via :func:`assign_teams` or
    :func:`assign_teams_from_slot_map`.
    """

    def make_round(prefix: str, slots: list[BracketSlot]) -> list[BracketSlot]:
        parents: list[BracketSlot] = []
        for i in range(0, len(slots), 2):
            parents.append(
                BracketSlot(
                    name=f"{prefix}_{i // 2}",
                    left=slots[i],
                    right=slots[i + 1],
                )
            )
        return parents

    leaves = [BracketSlot(name=f"seed{i + 1}") for i in range(32)]
    r16 = make_round("r16", leaves)
    qf = make_round("qf", r16)
    sf = make_round("sf", qf)
    final_pair = make_round("final", sf)
    champion = make_round("champion", final_pair)
    assert len(champion) == 1, f"expected 1 root, got {len(champion)}"
    return Bracket(root=champion[0])


# ---------------------------------------------------------------------------
# Legacy / demo assignment (sequential order)
# ---------------------------------------------------------------------------


def assign_teams(bracket: Bracket, seeding: list[str]) -> None:
    """Place 32 team names onto the bracket leaves in seed order.

    ``seeding[k]`` is assigned to the leaf named ``seed{k+1}``.
    Adjacent pairs (seed1↔seed2, seed3↔seed4, …) meet in the Round of 32.

    This is the *demo* path. For a real WC2026 forecast use
    :func:`assign_teams_from_slot_map` which honours the official pairings.
    """
    if len(seeding) != 32:
        raise ValueError(f"expected 32 teams, got {len(seeding)}")

    leaves: list[BracketSlot] = []

    def collect(node: BracketSlot) -> None:
        if node.is_leaf:
            leaves.append(node)
            return
        assert node.left and node.right
        collect(node.left)
        collect(node.right)

    collect(bracket.root)
    leaves.sort(key=lambda s: int(s.name.removeprefix("seed")))
    for slot, team in zip(leaves, seeding, strict=True):
        slot.team = team


# ---------------------------------------------------------------------------
# Slot-map types and official bracket ordering
# ---------------------------------------------------------------------------

# Maps official slot names to team names, e.g. {"1A": "Germany", "2B": "France", "3G": "Mexico"}.
SlotMap: TypeAlias = dict[str, str]

# Official R32 leaf order for WC2026.
# Each consecutive pair of entries meets in one Round of 32 match.
# Entry at position 2k (0-indexed) is the "home" side; 2k+1 is the "away" side.
#
# TODO(bracket-official): This ordering is a best-effort reconstruction from
# pre-tournament FIFA format documentation. Verify the exact 32-match slot
# assignments against the official FIFA 2026 match schedule before a production
# run. Incorrect pairings here produce wrong simulated Achtelfinale opponents.
#
# The slot names follow the convention:
#   "1X" = winner of group X (X in A..L)
#   "2X" = runner-up of group X
#   "3X" = qualifying third-placed team from group X
#
# Reference: https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026
WC2026_R32_SLOT_ORDER: list[str] = [
    # Left half of the bracket (Matches 49–56)
    "1A", "3rd_1",   # Match 49
    "1C", "3rd_2",   # Match 50
    "1B", "3rd_3",   # Match 51
    "1D", "3rd_4",   # Match 52
    "1E", "3rd_5",   # Match 53
    "1G", "3rd_6",   # Match 54
    "1F", "3rd_7",   # Match 55
    "1H", "3rd_8",   # Match 56
    # Right half of the bracket (Matches 57–64)
    "2A", "2B",   # Match 57
    "2C", "2D",   # Match 58
    "2E", "2F",   # Match 59
    "2G", "2H",   # Match 60
    "2I", "2J",   # Match 61
    "2K", "2L",   # Match 62
    "1I", "1J",   # Match 63
    "1K", "1L",   # Match 64
]

assert len(WC2026_R32_SLOT_ORDER) == 32, "slot order must have exactly 32 entries"


def assign_teams_from_slot_map(bracket: Bracket, slot_map: SlotMap) -> None:
    """Assign teams to the bracket using official WC2026 slot names.

    Maps slot names like ``"1A"`` (group A winner), ``"2B"`` (group B runner-up),
    or ``"3rd_1"`` (the 1st-ranked third-place qualifier) to team names, then
    places them onto the bracket leaves in the order defined by
    :data:`WC2026_R32_SLOT_ORDER`.

    Args:
        bracket: Empty bracket produced by :func:`build_empty_r32_bracket`.
        slot_map: Dict mapping every slot in ``WC2026_R32_SLOT_ORDER`` to a
            team name. Build it via :func:`build_slot_map_from_groups` or
            assemble it manually from the official FIFA bracket.

    Raises:
        ValueError: If any slot in the order is missing from ``slot_map``.
    """
    missing = [s for s in WC2026_R32_SLOT_ORDER if s not in slot_map]
    if missing:
        raise ValueError(
            f"slot_map is missing {len(missing)} slot(s): {missing}. "
            "Build it via build_slot_map_from_groups() or fill manually."
        )
    teams_in_bracket_order = [slot_map[s] for s in WC2026_R32_SLOT_ORDER]
    assign_teams(bracket, teams_in_bracket_order)


# ---------------------------------------------------------------------------
# Group results → slot map
# ---------------------------------------------------------------------------


@dataclass(order=False)
class GroupStanding:
    """One team's position in a group at the end of the group stage."""

    team: str
    group: str         # single letter, e.g. "A"
    points: int
    goal_diff: int
    goals_for: int
    goals_against: int = 0

    def _sort_key(self) -> tuple[int, int, int]:
        return (-self.points, -self.goal_diff, -self.goals_for)


def _rank_group(standings: list[GroupStanding]) -> list[GroupStanding]:
    """Sort group standings by FIFA criteria: points → GD → GF."""
    return sorted(standings, key=lambda s: s._sort_key())


def determine_third_place_qualifiers(
    group_standings: dict[str, list[GroupStanding]],
) -> list[GroupStanding]:
    """Return the 8 best third-placed teams by FIFA ranking criteria.

    Groups are ranked internally first (points → goal diff → goals for).
    The 3rd-place finisher of each group is then ranked across all 12 groups
    by the same criteria to determine the 8 qualifiers.

    Args:
        group_standings: Dict mapping group letter to its list of 4 standings.

    Returns:
        List of 8 qualifying third-placed GroupStanding objects, ranked best→worst.
    """
    thirds: list[GroupStanding] = []
    for group, standings in group_standings.items():
        ranked = _rank_group(standings)
        if len(ranked) >= 3:
            third = ranked[2]
            third.group = group
            thirds.append(third)

    return sorted(thirds, key=lambda s: s._sort_key())[:8]


def build_slot_map_from_groups(
    group_standings: dict[str, list[GroupStanding]],
) -> SlotMap:
    """Build a slot map from final group stage standings.

    Computes winners (``"1X"``), runners-up (``"2X"``), and the 8 best
    third-place qualifiers (``"3rd_1"``..``"3rd_8"``). The 3rd-place slots
    use a ranked-order key (``3rd_1`` = best, ``3rd_8`` = weakest) because
    the exact bracket position of each third-place team depends on the
    official FIFA contingency table which maps group-letter combinations to
    specific bracket slots.

    TODO(bracket-3rd-place): replace the ranked-order ``3rd_N`` keys with the
    official group-combination contingency slot names from the FIFA 2026
    regulations (Annex IV). This determines which R32 match each third-place
    team actually plays. Until then, a best-first ordering is used as a
    reasonable approximation.

    Args:
        group_standings: Dict mapping group letter ("A".."L") to its 4 standings.

    Returns:
        SlotMap ready to pass to :func:`assign_teams_from_slot_map`.
    """
    slot_map: SlotMap = {}

    for group, standings in group_standings.items():
        ranked = _rank_group(standings)
        if len(ranked) >= 1:
            slot_map[f"1{group}"] = ranked[0].team
        if len(ranked) >= 2:
            slot_map[f"2{group}"] = ranked[1].team

    qualifiers = determine_third_place_qualifiers(group_standings)
    for rank, standing in enumerate(qualifiers, start=1):
        slot_map[f"3rd_{rank}"] = standing.team

    return slot_map
