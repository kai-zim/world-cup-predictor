"""Automated reporting.

Generates human-readable Markdown reports from simulation output. Kept
dependency-light (pure string building) so it runs anywhere. Figure generation
(Plotly) lives in the dashboard; static report figures are a TODO extension.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def champion_probability_report(
    forecast: pd.DataFrame, stage_label: str, top_n: int = 16
) -> str:
    """Build a Markdown champion-probability report.

    ``forecast`` is the simulator output (team + p_* columns). ``stage_label``
    is e.g. "after_group_stage" / "after_round_of_16".
    """
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    cols = ["team", "p_round_of_16", "p_quarter_final", "p_semi_final", "p_final", "p_champion"]
    present = [c for c in cols if c in forecast.columns]
    table = forecast.sort_values("p_champion", ascending=False).head(top_n)[present]

    lines = [
        f"# Champion Probability Report — {stage_label}",
        "",
        f"_Generated: {ts}_",
        "",
        f"Top {top_n} teams by simulated title probability.",
        "",
        "| " + " | ".join(present) + " |",
        "| " + " | ".join(["---"] * len(present)) + " |",
    ]
    for row in table.itertuples(index=False):
        cells = []
        for c, v in zip(present, row, strict=True):
            cells.append(v if c == "team" else f"{v:.1%}")
        lines.append("| " + " | ".join(map(str, cells)) + " |")
    lines.append("")
    return "\n".join(lines)


def write_report(content: str, path: str | Path) -> Path:
    """Write a report to disk, creating parent dirs."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p
