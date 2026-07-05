"""Command-line entry point: ``worldcup-predictor <command>``.

Every command is runnable against the bundled example fixtures via
``--fixtures`` with no network access and no trained-model artifacts to
manage -- models are retrained from scratch on each invocation, which is
cheap at this data size (see README roadmap for artifact persistence).
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import pandas as pd

from worldcup_predictor.data.loaders import (
    HISTORICAL_MATCHES_FILENAME,
    download_historical,
    download_wc2026,
    load_raw_historical,
    load_raw_wc2026_from_dir,
)
from worldcup_predictor.data.preprocessing import build_historical_match_table, build_match_table
from worldcup_predictor.data.validation import missing_value_report
from worldcup_predictor.features.feature_pipeline import build_feature_frame, build_historical_feature_frame
from worldcup_predictor.models.baselines import HistoricalWinRateBaseline
from worldcup_predictor.models.evaluation import evaluate_predictions, run_time_based_backtest
from worldcup_predictor.models.predict import build_predictions_table
from worldcup_predictor.models.train import train_goal_model, train_outcome_model
from worldcup_predictor.reporting.reports import (
    data_quality_report,
    match_prediction_report,
    model_metrics_report,
    tournament_simulation_report,
)
from worldcup_predictor.simulation.probabilities import probability_table
from worldcup_predictor.simulation.tournament import simulate_tournament
from worldcup_predictor.utils.config import (
    PROJECT_ROOT,
    DataConfig,
    FeaturesConfig,
    ModelConfig,
    SimulationConfig,
    get_settings,
)
from worldcup_predictor.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)

FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures" / "wc2026"
HISTORICAL_FIXTURE = PROJECT_ROOT / "tests" / "fixtures" / "piterfm_sample.csv"

fixtures_option = click.option(
    "--fixtures/--no-fixtures",
    default=False,
    help="Use the bundled example data (tests/fixtures/wc2026) instead of a real download.",
)


@click.group()
@click.option("--log-level", default=None, help="Override WORLDCUP_LOG_LEVEL from .env")
def cli(log_level: str | None) -> None:
    settings = get_settings()
    configure_logging(log_level or settings.worldcup_log_level)


def _raw_dir(fixtures: bool) -> Path:
    if fixtures:
        return FIXTURES_DIR
    data_config = DataConfig.load()
    return PROJECT_ROOT / data_config.paths.raw_dir / "wc2026"


def _prepare(fixtures: bool) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    raw = load_raw_wc2026_from_dir(_raw_dir(fixtures))
    feature_frame = build_feature_frame(raw, FeaturesConfig.load())
    matches = build_match_table(raw)
    return raw, feature_frame, matches


def _train_models(played: pd.DataFrame, model_config: ModelConfig):
    if played.empty:
        raise click.ClickException(
            "No played matches available to train on -- check the raw data / --fixtures flag."
        )
    goal_model = train_goal_model(played, model_config.goal_model)
    outcome_model = train_outcome_model(played, model_config.outcome_model)
    historical_baseline = HistoricalWinRateBaseline.fit(played)
    return goal_model, outcome_model, historical_baseline


@cli.command("download-data")
@click.option("--source", type=click.Choice(["wc2026", "historical"]), default="wc2026")
@click.option("--force", is_flag=True, default=False, help="Re-download even if files already exist.")
def download_data(source: str, force: bool) -> None:
    """Download raw data into data/raw/ (see README for Kaggle credential setup)."""
    settings = get_settings()
    data_config = DataConfig.load()
    try:
        if source == "wc2026":
            paths = download_wc2026(data_config, PROJECT_ROOT, force=force)
            target_dir = PROJECT_ROOT / data_config.paths.raw_dir / "wc2026"
            click.echo(f"Downloaded {len(paths)} files to {target_dir}")
        else:
            target = download_historical(data_config, settings, PROJECT_ROOT)
            click.echo(f"Downloaded historical dataset to {target}")
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc


@cli.command("predict-matches")
@fixtures_option
def predict_matches(fixtures: bool) -> None:
    """Predict every match (baselines + goal model + outcome model), all matches side by side."""
    settings = get_settings()
    model_config = ModelConfig.load()
    raw, feature_frame, _ = _prepare(fixtures)

    played = feature_frame[feature_frame["winner"].notna()].copy()
    goal_model, outcome_model, historical_baseline = _train_models(played, model_config)

    predictions = build_predictions_table(feature_frame, goal_model, outcome_model, historical_baseline)

    output_dir = settings.worldcup_output_dir / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "match_predictions.csv", index=False)

    unplayed_ids = feature_frame.loc[feature_frame["winner"].isna(), "match_id"]
    upcoming = predictions[predictions["match_id"].isin(unplayed_ids)]
    report = match_prediction_report(upcoming if not upcoming.empty else predictions)
    reports_dir = settings.worldcup_output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "match_prediction_report.md").write_text(report, encoding="utf-8")

    metrics = {}
    model_names = ["outcome_model", "goal_model", "fifa_ranking", "elo_diff", "poisson", "historical"]
    if not played.empty:
        played_predictions = build_predictions_table(played, goal_model, outcome_model, historical_baseline)
        for model_name in model_names:
            prob_cols = [f"{model_name}_p_home", f"{model_name}_p_draw", f"{model_name}_p_away"]
            cols = played_predictions[prob_cols].rename(
                columns=lambda c, prefix=model_name: c.replace(f"{prefix}_", "")
            )
            metrics[model_name] = evaluate_predictions(played["winner"], cols)
    (reports_dir / "model_metrics_report.md").write_text(model_metrics_report(metrics), encoding="utf-8")

    dq_reports = {
        name: missing_value_report(df, name)
        for name, df in raw.items()
    }
    (reports_dir / "data_quality_report.md").write_text(data_quality_report(dq_reports), encoding="utf-8")

    click.echo(f"Wrote predictions for {len(predictions)} matches to {output_dir}")
    click.echo(f"Reports written to {reports_dir}")


@cli.command("simulate-tournament")
@fixtures_option
@click.option("--n-simulations", type=int, default=None, help="Override configs/simulation.yaml")
def simulate_tournament_cmd(fixtures: bool, n_simulations: int | None) -> None:
    """Monte Carlo simulate the remaining knockout bracket."""
    settings = get_settings()
    model_config = ModelConfig.load()
    sim_config = SimulationConfig.load()
    if n_simulations:
        sim_config.monte_carlo.n_simulations = n_simulations

    raw, feature_frame, matches = _prepare(fixtures)
    played = feature_frame[feature_frame["winner"].notna()].copy()
    goal_model, _, _ = _train_models(played, model_config)

    result = simulate_tournament(matches, feature_frame, goal_model, sim_config)
    table = probability_table(result, sim_config, raw["teams"])

    sim_dir = settings.worldcup_output_dir / "simulations"
    sim_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(sim_dir / "tournament_probabilities.csv", index=False)
    (sim_dir / "tournament_probabilities.json").write_text(
        json.dumps(
            {"n_simulations": result.n_simulations, "teams": table.to_dict(orient="records")}, indent=2
        ),
        encoding="utf-8",
    )

    reports_dir = settings.worldcup_output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report = tournament_simulation_report(table, result.n_simulations)
    (reports_dir / "tournament_simulation_report.md").write_text(report, encoding="utf-8")

    click.echo(f"Simulated {result.n_simulations:,} tournaments. Champion favorite: {table.iloc[0]['team']}")
    click.echo(f"Wrote {sim_dir / 'tournament_probabilities.csv'} and the simulation report to {reports_dir}")


@cli.command("backtest-historical")
@fixtures_option
def backtest_historical(fixtures: bool) -> None:
    """Time-based backtest (train-until-X -> test-year-Y) on the 1930-2022 historical dataset.

    Uses a reduced feature set (Elo + rolling form + rest days only -- see
    configs/model.yaml: historical_backtest) since squad market value and
    per-match FIFA ranking aren't available that far back.
    """
    settings = get_settings()
    features_config = FeaturesConfig.load()
    model_config = ModelConfig.load()

    if fixtures:
        historical_path = HISTORICAL_FIXTURE
    else:
        data_config = DataConfig.load()
        historical_dir = PROJECT_ROOT / data_config.paths.raw_dir / "historical"
        historical_path = historical_dir / HISTORICAL_MATCHES_FILENAME

    raw_historical = load_raw_historical(historical_path)
    historical_matches = build_historical_match_table(raw_historical)
    historical_features = build_historical_feature_frame(historical_matches, features_config)

    results = run_time_based_backtest(historical_features, model_config)
    if results.empty:
        raise click.ClickException(
            "No backtest splits produced results -- none of configs/model.yaml's "
            "evaluation.time_based_splits had played matches in both the train and "
            "test window of the loaded historical data."
        )

    reports_dir = settings.worldcup_output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(reports_dir / "historical_backtest_results.csv", index=False)

    lines = [
        "# Historical Backtest Report\n",
        f"{len(historical_matches)} historical matches (1930-2022), reduced feature set "
        "(Elo + rolling form + rest days only).\n",
    ]
    for split_label, group in results.groupby("split"):
        lines.append(f"## {split_label}\n")
        lines.append(group.drop(columns=["split"]).round(4).to_markdown(index=False))
        lines.append("")
    (reports_dir / "historical_backtest_report.md").write_text("\n".join(lines), encoding="utf-8")

    click.echo(f"Ran {results['split'].nunique()} backtest split(s); wrote results to {reports_dir}")


@cli.command("update-after-round")
@fixtures_option
@click.option("--n-simulations", type=int, default=None)
@click.pass_context
def update_after_round(ctx: click.Context, fixtures: bool, n_simulations: int | None) -> None:
    """Re-run the full pipeline: predictions + simulation + all reports.

    Call this after every completed knockout round (new results downloaded).
    """
    ctx.invoke(predict_matches, fixtures=fixtures)
    ctx.invoke(simulate_tournament_cmd, fixtures=fixtures, n_simulations=n_simulations)


if __name__ == "__main__":
    cli()
