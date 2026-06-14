"""Tests for bracket construction and slot-map assignment (wc2026.simulation.bracket)."""

from __future__ import annotations

import pytest

from wc2026.simulation.bracket import (
    GroupStanding,
    WC2026_R32_SLOT_ORDER,
    assign_teams,
    assign_teams_from_slot_map,
    build_empty_r32_bracket,
    build_slot_map_from_groups,
    determine_third_place_qualifiers,
)


# ── slot order invariants ──────────────────────────────────────────────────────


def test_slot_order_has_32_entries() -> None:
    assert len(WC2026_R32_SLOT_ORDER) == 32


def test_slot_order_no_duplicates() -> None:
    assert len(set(WC2026_R32_SLOT_ORDER)) == 32


# ── assign_teams_from_slot_map ─────────────────────────────────────────────────


def _full_slot_map() -> dict[str, str]:
    """Build a complete slot map aligned with WC2026_R32_SLOT_ORDER."""
    return {slot: f"Team_{slot}" for slot in WC2026_R32_SLOT_ORDER}


def test_assign_from_slot_map_fills_all_leaves() -> None:
    bracket = build_empty_r32_bracket()
    assign_teams_from_slot_map(bracket, _full_slot_map())

    leaves: list = []

    def collect(node):  # type: ignore[no-untyped-def]
        if node.is_leaf:
            leaves.append(node)
        else:
            collect(node.left)
            collect(node.right)

    collect(bracket.root)
    assert all(leaf.team is not None for leaf in leaves)
    assert len(leaves) == 32


def test_assign_from_slot_map_missing_slot_raises() -> None:
    bracket = build_empty_r32_bracket()
    incomplete = {slot: f"T_{slot}" for slot in WC2026_R32_SLOT_ORDER[:-1]}  # one missing
    with pytest.raises(ValueError, match="missing"):
        assign_teams_from_slot_map(bracket, incomplete)


# ── GroupStanding & third-place qualifier ─────────────────────────────────────


def _make_group(group: str, standings: list[tuple[str, int, int, int]]) -> list[GroupStanding]:
    """Helper: list of (team, points, goal_diff, goals_for)."""
    return [
        GroupStanding(team=t, group=group, points=p, goal_diff=gd, goals_for=gf, goals_against=0)
        for t, p, gd, gf in standings
    ]


def test_determine_third_place_qualifiers_returns_8() -> None:
    groups = {}
    for i, letter in enumerate("ABCDEFGHIJKL"):
        groups[letter] = _make_group(letter, [
            (f"W{letter}", 9, 6, 8),
            (f"R{letter}", 6, 2, 5),
            (f"T{letter}", 3 - i % 3, -1 - i, 2),
            (f"L{letter}", 0, -7, 1),
        ])
    qualifiers = determine_third_place_qualifiers(groups)
    assert len(qualifiers) == 8


def test_third_place_ranked_by_points_then_gd() -> None:
    groups = {
        "A": _make_group("A", [("A1", 9, 5, 8), ("A2", 6, 2, 5), ("A3", 4, 1, 3), ("A4", 0, -8, 0)]),
        "B": _make_group("B", [("B1", 9, 4, 7), ("B2", 6, 1, 4), ("B3", 2, -2, 2), ("B4", 0, -3, 1)]),
    }
    # Only 2 groups, so only 2 thirds — best should be A3 (4 pts > 2 pts)
    qualifiers = determine_third_place_qualifiers(groups)
    assert len(qualifiers) == 2
    assert qualifiers[0].team == "A3"
    assert qualifiers[1].team == "B3"


def test_build_slot_map_from_groups_contains_all_slots() -> None:
    groups = {}
    for letter in "ABCDEFGHIJKL":
        groups[letter] = _make_group(letter, [
            (f"W{letter}", 9, 5, 7),
            (f"R{letter}", 6, 1, 4),
            (f"T{letter}", 1, -3, 2),
            (f"L{letter}", 0, -3, 1),
        ])
    slot_map = build_slot_map_from_groups(groups)

    # 12 winners + 12 runners-up + 8 third-place = 32 slots
    assert len([k for k in slot_map if k.startswith("1")]) == 12
    assert len([k for k in slot_map if k.startswith("2")]) == 12
    assert len([k for k in slot_map if k.startswith("3rd_")]) == 8


def test_build_slot_map_winner_is_highest_points() -> None:
    groups = {
        "A": _make_group("A", [
            ("Germany", 9, 6, 8),
            ("France", 6, 2, 5),
            ("Portugal", 3, -3, 3),
            ("Chile", 0, -5, 1),
        ])
    }
    slot_map = build_slot_map_from_groups(groups)
    assert slot_map["1A"] == "Germany"
    assert slot_map["2A"] == "France"


# ── legacy assign_teams ────────────────────────────────────────────────────────


def test_assign_teams_requires_32() -> None:
    bracket = build_empty_r32_bracket()
    with pytest.raises(ValueError, match="32"):
        assign_teams(bracket, ["only", "two"])


def test_assign_teams_sequential_fills_leaves() -> None:
    bracket = build_empty_r32_bracket()
    teams = [f"T{i:02d}" for i in range(32)]
    assign_teams(bracket, teams)

    leaves = []

    def collect(n):  # type: ignore[no-untyped-def]
        if n.is_leaf:
            leaves.append(n)
        else:
            collect(n.left)
            collect(n.right)

    collect(bracket.root)
    assert all(leaf.team is not None for leaf in leaves)
