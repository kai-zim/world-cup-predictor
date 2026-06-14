"""Command-line interface (Typer).

Subcommands:
    wc2026 forecast   --history <csv> --seeding <csv> [--config <yaml>]
    wc2026 calibrate  --history <csv> [--config <yaml>] [--out <yaml>]
    wc2026 demo       [--n <int>]

``demo`` runs without any data files and is the fastest way to verify the install.
``calibrate`` fits the EloPoissonModel constants on real data and prints the results.
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
    seeding: Path = typer.Option(
        ..., help="CSV with a single column 'team' listing 32 qualified teams in bracket slot order."
    ),
    config: Path | None = typer.Option(None, help="YAML config; uses defaults if omitted."),
    out: Path = typer.Option(
        Path("outputs/reports/champion_probabilities.md"),
        help="Output path for the Markdown report.",
    ),
    stage: str = typer.Option("after_group_stage", help="Stage label used in the report header."),
) -> None:
    """Run a full Monte-Carlo knockout forecast from real data files."""
    configure_logging()
    cfg = _load_config(config)
    hist = pd.read_csv(history, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    teams = pd.read_csv(seeding)["team"].tolist()
    fc, ratings, _bracket = run_forecast(hist, teams, cfg)
    report = champion_probability_report(fc, stage_label=stage)
    path = write_report(report, out)
    typer.echo(f"Wrote report → {path}")
    typer.echo(f"\nTop 5 by title probability:\n{fc.head(5)[['team', 'p_champion']].to_string(index=False)}")


@app.command()
def calibrate(
    history: Path = typer.Option(..., help="CSV of historical results used for calibration."),
    config: Path | None = typer.Option(None, help="YAML config; uses defaults if omitted."),
    train_start: str = typer.Option("2022-12-19", help="Start of training window (ISO date)."),
    train_end: str | None = typer.Option(None, help="End of training window (ISO date). Defaults to today."),
    out: Path | None = typer.Option(None, help="Optional path to write fitted constants as YAML snippet."),
) -> None:
    """Fit EloPoissonModel constants via MLE on historical match data.

    Prints the fitted base_rate, supremacy_scale, and home_advantage_goals.
    Copy the values into configs/default.yaml (under the ``poisson:`` block or
    a new ``elo_poisson:`` section) to use them in subsequent forecasts.
    """
    configure_logging()
    cfg = _load_config(config)

    from wc2026.data.loaders import filter_window, load_results
    from wc2026.data.validate_data import validate_match_frame
    from wc2026.features.engineering import assemble_match_features
    from wc2026.models.calibrate import calibrate_elo_poisson
    from wc2026.models.elo import EloEngine

    typer.echo("Loading and validating results …")
    df = load_results(history)
    validate_match_frame(df)

    end = train_end or str(pd.Timestamp.today().normalize().date())
    df = filter_window(df, start=train_start, end=end)
    typer.echo(f"Training window: {train_start} → {end}  ({len(df):,} matches)")

    typer.echo("Replaying Elo …")
    engine = EloEngine(cfg.elo)
    df_elo = engine.replay(df)

    typer.echo("Building features …")
    features = assemble_match_features(df_elo)

    typer.echo("Optimising parameters (Nelder-Mead) …")
    model = calibrate_elo_poisson(features)

    typer.echo("\n── Calibrated EloPoissonModel constants ──")
    typer.echo(f"  base_rate            = {model._base:.4f}")
    typer.echo(f"  supremacy_scale      = {model._scale:.6f}")
    typer.echo(f"  home_advantage_goals = {model._home_goal_adv:.4f}")

    snippet = (
        "# Paste these into configs/default.yaml:\n"
        "elo_poisson:\n"
        f"  base_rate: {model._base:.4f}\n"
        f"  supremacy_scale: {model._scale:.6f}\n"
        f"  home_advantage_goals: {model._home_goal_adv:.4f}\n"
    )
    typer.echo(f"\n{snippet}")

    if out is not None:
        out.write_text(snippet, encoding="utf-8")
        typer.echo(f"Wrote constants → {out}")


@app.command()
def demo(n: int = typer.Option(5000, help="Number of Monte-Carlo simulations.")) -> None:
    """Synthetic end-to-end demonstration — no external data files required."""
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
                date=d,
                tournament="Friendly",
                home_team=a,
                away_team=b,
                home_score=int(rng.poisson(max(0.3, 2.0 - sa * 0.05))),
                away_score=int(rng.poisson(max(0.3, 2.0 - sb * 0.05))),
                neutral=True,
            )
        )
    hist = pd.DataFrame(rows)
    fc, _r, _b = run_forecast(hist, teams, cfg)
    typer.echo(f"── Demo forecast ({n:,} simulations) ──")
    typer.echo(fc.head(10).to_string(index=False))


if __name__ == "__main__":
    app()
