"""Typed configuration loading.

Every tunable in this project lives in configs/*.yaml, never hardcoded in
source files. This module turns those YAML files into validated Pydantic
models, so a malformed config fails fast with a clear error instead of
silently producing wrong features or simulations.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIGS_DIR = PROJECT_ROOT / "configs"


class Settings(BaseSettings):
    """Environment-driven settings (see .env.example). Everything is optional
    for the MVP: tests and the CLI run against fixtures without any of this set."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"), env_file_encoding="utf-8", extra="ignore"
    )

    kaggle_username: str | None = None
    kaggle_key: str | None = None
    worldcup_data_dir: Path = PROJECT_ROOT / "data"
    worldcup_output_dir: Path = PROJECT_ROOT / "outputs"
    worldcup_log_level: str = "INFO"

    @model_validator(mode="after")
    def _resolve_relative_paths(self) -> Settings:
        if not self.worldcup_data_dir.is_absolute():
            self.worldcup_data_dir = PROJECT_ROOT / self.worldcup_data_dir
        if not self.worldcup_output_dir.is_absolute():
            self.worldcup_output_dir = PROJECT_ROOT / self.worldcup_output_dir
        return self

    def has_kaggle_credentials(self) -> bool:
        placeholders = {"PLATZHALTER_KAGGLE_USERNAME", "PLATZHALTER_KAGGLE_KEY"}
        return bool(
            self.kaggle_username
            and self.kaggle_key
            and self.kaggle_username not in placeholders
            and self.kaggle_key not in placeholders
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()


def _load_yaml(name: str) -> dict:
    path = CONFIGS_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"Config file {path} not found. Expected it next to pyproject.toml under configs/."
        )
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        raise ValueError(f"Config file {path} is empty.")
    return data


# --- data.yaml -----------------------------------------------------------------


class DataPaths(BaseModel):
    raw_dir: str
    interim_dir: str
    processed_dir: str
    external_dir: str


class DataSourceConfig(BaseModel):
    description: str
    kind: Literal["github_raw", "kaggle"]
    license: str
    repo: str | None = None
    branch: str | None = None
    files: list[str] = Field(default_factory=list)
    dataset: str | None = None
    required_env: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_kind_fields(self) -> DataSourceConfig:
        if self.kind == "github_raw" and not (self.repo and self.branch and self.files):
            raise ValueError("github_raw sources require repo, branch and files")
        if self.kind == "kaggle" and not self.dataset:
            raise ValueError("kaggle sources require a dataset slug")
        return self


class DataConfig(BaseModel):
    paths: DataPaths
    sources: dict[str, DataSourceConfig]
    optional_sources: list[str] = Field(default_factory=list)

    @classmethod
    def load(cls) -> DataConfig:
        return cls.model_validate(_load_yaml("data.yaml"))


# --- features.yaml ---------------------------------------------------------


class EloConfig(BaseModel):
    initial_rating: float
    k_factor_group_stage: float
    k_factor_knockout: float
    k_factor_friendly: float
    home_advantage: float


class RollingFormConfig(BaseModel):
    windows: list[int]
    metrics: list[str]


class RestDaysConfig(BaseModel):
    cap_days: int


class SquadFeaturesConfig(BaseModel):
    top_n_market_value: int
    caps_threshold: int


class GroupStageFeaturesConfig(BaseModel):
    enabled: bool


class VenueFeaturesConfig(BaseModel):
    use_altitude: bool


class FeaturesConfig(BaseModel):
    elo: EloConfig
    rolling_form: RollingFormConfig
    rest_days: RestDaysConfig
    squad_features: SquadFeaturesConfig
    group_stage_features: GroupStageFeaturesConfig
    venue_features: VenueFeaturesConfig

    @classmethod
    def load(cls) -> FeaturesConfig:
        return cls.model_validate(_load_yaml("features.yaml"))


# --- model.yaml --------------------------------------------------------------


class GoalModelConfig(BaseModel):
    kind: str
    features: list[str]
    regularization_alpha: float


class OutcomeModelConfig(BaseModel):
    kind: str
    features: list[str]
    params: dict


class CalibrationConfig(BaseModel):
    method: Literal["isotonic", "platt"]


class TimeSplit(BaseModel):
    train_until: int
    test_year: int


class EvaluationConfig(BaseModel):
    metrics: list[str]
    time_based_splits: list[TimeSplit]


class HistoricalBacktestConfig(BaseModel):
    """Reduced feature lists for the 1930-2022 historical backtest -- squad
    market value and per-match FIFA ranking are not available that far back,
    unlike the wc2026 goal_model/outcome_model feature lists above."""

    goal_model_features: list[str]
    outcome_model_features: list[str]


class ModelConfig(BaseModel):
    goal_model: GoalModelConfig
    outcome_model: OutcomeModelConfig
    baselines: list[str]
    calibration: CalibrationConfig
    evaluation: EvaluationConfig
    historical_backtest: HistoricalBacktestConfig

    @classmethod
    def load(cls) -> ModelConfig:
        return cls.model_validate(_load_yaml("model.yaml"))


# --- simulation.yaml -----------------------------------------------------------


class MonteCarloConfig(BaseModel):
    n_simulations: int
    random_seed: int


class KnockoutConfig(BaseModel):
    penalty_shootout_home_win_prob: float
    extra_time_goal_rate_multiplier: float


class StagesConfig(BaseModel):
    order: list[str]


class SimulationReportingConfig(BaseModel):
    probability_decimal_places: int


class SimulationConfig(BaseModel):
    monte_carlo: MonteCarloConfig
    knockout: KnockoutConfig
    stages: StagesConfig
    reporting: SimulationReportingConfig

    @classmethod
    def load(cls) -> SimulationConfig:
        return cls.model_validate(_load_yaml("simulation.yaml"))
