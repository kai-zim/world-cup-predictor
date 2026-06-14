"""World Cup 2026 knockout bracket definition.

WC2026 introduces a 48-team format: 12 groups of 4. The 12 group winners, 12
runners-up, plus the 8 best third-placed teams advance to a 32-team knockout
(Round of 32 -> R16 -> QF -> SF -> Final). This module encodes the bracket
*topology* as a binary tree of slots; concrete team assignments are injected
once the group stage finishes.

The bracket pairing below follows the published FIFA 2026 slot map shape. The
exact slot-to-group mapping (e.g. which group's winner meets which) is a fixed
lookup that should be verified against the official match schedule before a
production run — flagged with TODO(bracket).
"""

from __future__ import annotations

from dataclasses import dataclass, field


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
    # Ordered round labels from first knockout round to final.
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

    Leaves are named seed1..seed32 in bracket order (adjacent pairs meet in the
    Round of 32). Team assignment happens in
    :func:`assign_teams`.
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
    r16 = make_round("r16", leaves)  # 16 winners of R32 ties
    qf = make_round("qf", r16)  # 8
    sf = make_round("sf", qf)  # 4
    final_pair = make_round("final", sf)  # 2 -> meet in the final
    champion = make_round("champion", final_pair)  # 1 root
    assert len(champion) == 1, f"expected 1 root, got {len(champion)}"
    return Bracket(root=champion[0])


def assign_teams(bracket: Bracket, seeding: list[str]) -> None:
    """Place 32 team names onto the bracket leaves in seed order.

    ``seeding[k]`` is assigned to the leaf named ``seed{k+1}``.

    TODO(bracket): replace the naive sequential seeding with the official FIFA
    2026 slot map (group winners/runners-up/3rd-place placement) so that the
    simulated pairings match the real tournament path.
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
