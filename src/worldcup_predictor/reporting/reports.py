"""Markdown/CSV report generation.

Every report is a pure function: DataFrame(s) in, markdown string out. The
CLI decides where to write it (outputs/reports/) and whether to also dump
the underlying CSV -- these functions never touch the filesystem themselves.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd


def _timestamp() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M")


def match_prediction_report(predictions: pd.DataFrame) -> str:
    lines = [f"# Match Prediction Report\n\nGenerated: {_timestamp()}\n"]
    for _, row in predictions.iterrows():
        lines.append(f"## {row['home_team']} vs {row['away_team']} ({row['stage']}, {row['date']})\n")
        comparison = pd.DataFrame(
            {
                "model": [
                    "Outcome model (LightGBM)",
                    "Goal model (Poisson)",
                    "FIFA ranking baseline",
                    "Elo baseline",
                    "Poisson baseline",
                    "Historical win-rate baseline",
                ],
                "p_home": [
                    row["outcome_model_p_home"], row["goal_model_p_home"], row["fifa_ranking_p_home"],
                    row["elo_diff_p_home"], row["poisson_p_home"], row["historical_p_home"],
                ],
                "p_draw": [
                    row["outcome_model_p_draw"], row["goal_model_p_draw"], row["fifa_ranking_p_draw"],
                    row["elo_diff_p_draw"], row["poisson_p_draw"], row["historical_p_draw"],
                ],
                "p_away": [
                    row["outcome_model_p_away"], row["goal_model_p_away"], row["fifa_ranking_p_away"],
                    row["elo_diff_p_away"], row["poisson_p_away"], row["historical_p_away"],
                ],
            }
        ).round(3)
        lines.append(comparison.to_markdown(index=False))
        lines.append(
            f"\nExpected goals (goal model): {row['home_team']} {row['goal_model_expected_home_goals']:.2f} "
            f"- {row['goal_model_expected_away_goals']:.2f} {row['away_team']}\n"
        )
    return "\n".join(lines)


def tournament_simulation_report(probability_table: pd.DataFrame, n_simulations: int) -> str:
    lines = [
        "# Tournament Simulation Report\n",
        f"Generated: {_timestamp()}  \nMonte Carlo draws: {n_simulations:,}\n",
        "## Champion probability\n",
    ]
    champion_cols = ["team", "prob_champion", "prob_final", "prob_semi_final"]
    available = [c for c in champion_cols if c in probability_table.columns]
    lines.append(probability_table[available].to_markdown(index=False))
    return "\n".join(lines)


def data_quality_report(missing_reports: dict[str, pd.DataFrame]) -> str:
    lines = [f"# Data Quality Report\n\nGenerated: {_timestamp()}\n"]
    for name, report in missing_reports.items():
        lines.append(f"## {name}\n")
        nonzero = report[report["missing_count"] > 0]
        if nonzero.empty:
            lines.append("No missing values.\n")
        else:
            lines.append(nonzero.to_markdown(index=False))
            lines.append("")
    return "\n".join(lines)


def model_metrics_report(metrics_by_model: dict[str, dict[str, float]]) -> str:
    lines = [f"# Model Metrics Report\n\nGenerated: {_timestamp()}\n"]
    table = pd.DataFrame(metrics_by_model).T.round(4)
    table.index.name = "model"
    lines.append(table.reset_index().to_markdown(index=False))
    return "\n".join(lines)
