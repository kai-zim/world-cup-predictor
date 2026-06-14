"""Command-line interface (Typer).

Subcommands:
    wc2026 forecast --config configs/default.yaml --history <csv> --seeding <csv>
    wc2026 demo       # synthetic end-to-end run, no data needed

The ``demo`` command is the fastest way to verify an install works.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import typer

from wc2026.config.schema import AppConfig
from wc2026.pipeline import run_forecast
from wc2026.reporting.reports import champion_probability_report, write_report
from wc2026.utils.logging import configure_logging, get_logger

app = typer.Typer(add_completion=False, help="WC2026 knockout-stage predictor")
log = get_logger(__name__)


def _load_config(config: Path | None) -> AppConfig:
    return AppConfig.from_yaml(config) if config else AppConfig()


@app.command()
def forecast(
    history: Path = typer.Option(..., help="CSV of historical results (normalised schema)."),
    seeding: Path = typer.Option(..., help="CSV with a single column 'team' of 32 qualified teams in seed order."),
    config: Path | None = typer.Option(None, help="YAML config; defaults if omitted."),
    out: Path = typer.Option(Path("outputs/reports/champion_probabilities.md")),
) -> None:
    """Run a forecast from real data files."""
    configure_logging()
    cfg = _load_config(config)
    hist = pd.read_csv(history, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    teams = pd.read_csv(seeding)["team"].tolist()
    fc, _ratings, _bracket = run_forecast(hist, teams, cfg)
    report = champion_probability_report(fc, stage_label="after_group_stage")
    path = write_report(report, out)
    typer.echo(f"Wrote {path}")


@app.command()
def demo(n: int = typer.Option(5000, help="Monte-Carlo simulations.")) -> None:
    """Synthetic end-to-end demonstration (no external data required)."""
    configure_logging()
    cfg = AppConfig()
    cfg.simulation.n_simulations = n

    rng = np.random.default_rng(0)
    teams = [f"Team{i:02d}" for i in range(32)]
    dates = pd.date_range("2022-12-19", periods=500, freq="3D")
    rows = []
    for d in dates:
        a, b = rng.choice(teams, 2, replace=False)
        sa, sb = int(a[4:]), int(b[4:])
        rows.append(
            dict(
                date=d, tournament="Friendly", home_team=a, away_team=b,
                home_score=int(rng.poisson(max(0.3, 2.0 - sa * 0.05))),
                away_score=int(rng.poisson(max(0.3, 2.0 - sb * 0.05))),
                neutral=True,
            )
        )
    hist = pd.DataFrame(rows)
    fc, _r, _b = run_forecast(hist, teams, cfg)
    typer.echo(fc.head(10).to_string(index=False))


if __name__ == "__main__":
    app()
