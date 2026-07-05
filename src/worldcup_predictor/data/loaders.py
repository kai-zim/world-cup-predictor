"""Download and load raw data.

Raw data is never committed to the repo (see .gitignore) and never modified
in place -- this module only ever writes into ``data/raw/<source>/`` and
always keeps the provider's original column names and values.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from worldcup_predictor.data.schemas import (
    HistoricalMatchesRawSchema,
    MatchesRawSchema,
    MatchTeamStatsRawSchema,
    SquadsAndPlayersRawSchema,
    TeamsRawSchema,
    TournamentStagesRawSchema,
    VenuesRawSchema,
)
from worldcup_predictor.utils.config import DataConfig, Settings
from worldcup_predictor.utils.logging import get_logger

logger = get_logger(__name__)

GITHUB_RAW_URL = "https://raw.githubusercontent.com/{repo}/{branch}/{file}"

_WC2026_SCHEMAS = {
    "teams.csv": TeamsRawSchema,
    "venues.csv": VenuesRawSchema,
    "tournament_stages.csv": TournamentStagesRawSchema,
    "matches.csv": MatchesRawSchema,
    "squads_and_players.csv": SquadsAndPlayersRawSchema,
    "match_team_stats.csv": MatchTeamStatsRawSchema,
}


def download_wc2026(config: DataConfig, project_root: Path, force: bool = False) -> dict[str, Path]:
    """Download every configured file of the mominullptr/FIFA-World-Cup-2026-Dataset.

    Returns a mapping of filename -> local path. Existing files are kept
    unless ``force=True``, so re-running this is cheap and safe.
    """
    source = config.sources["wc2026"]
    target_dir = project_root / config.paths.raw_dir / "wc2026"
    target_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    for filename in source.files:
        target = target_dir / filename
        if target.exists() and not force:
            logger.debug("Skipping %s, already downloaded", filename)
            paths[filename] = target
            continue

        url = GITHUB_RAW_URL.format(repo=source.repo, branch=source.branch, file=filename)
        logger.info("Downloading %s", url)
        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to download {filename} from {url}: HTTP {response.status_code}. "
                "The upstream repo may have renamed/moved this file -- check "
                "configs/data.yaml against the current repo contents."
            )
        target.write_bytes(response.content)
        paths[filename] = target

    return paths


def download_historical(config: DataConfig, settings: Settings, project_root: Path) -> Path:
    """Download the piterfm/fifa-football-world-cup Kaggle dataset.

    Requires KAGGLE_USERNAME and KAGGLE_KEY (see .env.example). Raises a clear
    error instead of silently falling back to fixture/synthetic data -- the
    caller decides whether to use fixtures for local dev/tests.
    """
    if not settings.has_kaggle_credentials():
        raise RuntimeError(
            "Kaggle credentials not configured. Set KAGGLE_USERNAME and KAGGLE_KEY "
            "in your .env (see .env.example), then re-run. Get a token at "
            "https://www.kaggle.com/settings -> Create New Token. Until then, "
            "historical backtesting and the piterfm-based features stay disabled; "
            "tests use tests/fixtures/piterfm_sample.csv instead."
        )

    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError(
            "The 'kagglehub' package is required to download Kaggle datasets. "
            "Install it with: uv pip install worldcup-predictor[kaggle]"
        ) from exc

    import os

    os.environ.setdefault("KAGGLE_USERNAME", settings.kaggle_username or "")
    os.environ.setdefault("KAGGLE_KEY", settings.kaggle_key or "")

    source = config.sources["historical"]
    logger.info("Downloading Kaggle dataset %s", source.dataset)
    cache_path = Path(kagglehub.dataset_download(source.dataset))

    target_dir = project_root / config.paths.raw_dir / "historical"
    target_dir.mkdir(parents=True, exist_ok=True)
    for csv_file in cache_path.glob("*.csv"):
        target = target_dir / csv_file.name
        target.write_bytes(csv_file.read_bytes())
    return target_dir


def load_raw_wc2026_from_dir(base_dir: Path) -> dict[str, pd.DataFrame]:
    """Load and schema-validate every wc2026 raw CSV from an arbitrary directory.

    Used both for the real downloaded data (data/raw/wc2026) and for the
    bundled example fixtures (tests/fixtures/wc2026), so the CLI and the
    dashboard can run against small example data without any network access.
    """
    frames: dict[str, pd.DataFrame] = {}
    for filename, schema in _WC2026_SCHEMAS.items():
        path = base_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run the download step first "
                "(worldcup-predictor download-data), or pass --fixtures to use "
                "the bundled example data."
            )
        df = pd.read_csv(path)
        try:
            df = schema.validate(df, lazy=True)
        except Exception as exc:  # pandera.errors.SchemaErrors
            raise ValueError(
                f"{filename} failed schema validation -- the upstream dataset "
                f"structure may have changed. Details:\n{exc}"
            ) from exc
        frames[filename.removesuffix(".csv")] = df
    return frames


def load_raw_wc2026(config: DataConfig, project_root: Path) -> dict[str, pd.DataFrame]:
    """Load and schema-validate every wc2026 raw CSV that has been downloaded."""
    raw_dir = project_root / config.paths.raw_dir / "wc2026"
    return load_raw_wc2026_from_dir(raw_dir)


HISTORICAL_MATCHES_FILENAME = "matches_1930_2022.csv"


def load_raw_historical(path: Path) -> pd.DataFrame:
    """Load and schema-validate the historical match-level file (real download or fixture)."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")
    df = pd.read_csv(path)
    try:
        df = HistoricalMatchesRawSchema.validate(df, lazy=True)
    except Exception as exc:
        raise ValueError(
            f"{path} failed schema validation against HistoricalMatchesRawSchema "
            "(src/worldcup_predictor/data/schemas.py) -- this schema was verified "
            f"against a real download on 2026-07-05, so a mismatch now likely means "
            f"the provider changed the file. Details:\n{exc}"
        ) from exc
    return df


def load_raw_historical_from_dir(base_dir: Path) -> pd.DataFrame:
    """Load the match-level historical file from a downloaded historical/ directory.

    The Kaggle dataset ships 5 files total (2 FIFA ranking snapshots, the 2026
    schedule, a per-tournament summary, and this match-level file); only the
    match-level file is wired into the pipeline so far -- see README roadmap.
    """
    return load_raw_historical(base_dir / HISTORICAL_MATCHES_FILENAME)
