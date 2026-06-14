"""Typed configuration schema for the WC2026 predictor.

Configuration is loaded from YAML (``configs/*.yaml``) and validated through
Pydantic. Every pipeline stage receives a fully-typed config object, so that
invalid setups fail fast at load time rather than deep inside a training run.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator


class PathsConfig(BaseModel):
    """Filesystem layout. Relative paths are resolved against the repo root."""

    raw: Path = Path("data/raw")
    interim: Path = Path("data/interim")
    processed: Path = Path("data/processed")
    external: Path = Path("data/external")
    models: Path = Path("models")
    outputs: Path = Path("outputs")


class EloConfig(BaseModel):
    """Parameters for the international Elo rating system.

    ``k_base`` follows the World Football Elo convention; the effective K is
    scaled by tournament importance and goal-difference multipliers.
    """

    start_rating: float = 1500.0
    k_base: float = 40.0
    home_advantage: float = 65.0
    # Tournament importance weights (eloratings.net convention).
    importance: dict[str, float] = Field(
        default_factory=lambda: {
            "FIFA World Cup": 60.0,
            "UEFA Euro": 50.0,
            "Copa America": 50.0,
            "African Cup of Nations": 50.0,
            "AFC Asian Cup": 50.0,
            "Gold Cup": 50.0,
            "UEFA Nations League": 40.0,
            "FIFA World Cup qualification": 40.0,
            "UEFA Euro qualification": 40.0,
            "Friendly": 20.0,
        }
    )
    default_importance: float = 30.0


class PoissonConfig(BaseModel):
    """Goal model + match resolution settings."""

    max_goals: int = 10  # truncation for the scoreline grid
    # Probability of the higher-Elo side winning a penalty shootout when a
    # knockout match is level after extra time. 0.5 == coin flip.
    shootout_elo_scale: float = 0.0025


class SimulationConfig(BaseModel):
    n_simulations: int = 100_000
    random_seed: int = 42

    @field_validator("n_simulations")
    @classmethod
    def _positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("n_simulations must be > 0")
        return v


class ModelConfig(BaseModel):
    """Selection + hyperparameters for the optional learned goal model.

    The default production path uses the analytic Elo->Poisson mapping. When
    ``backend`` is set to a learned model, training data with the required
    feature columns must be available (see features/ TODO modules).
    """

    backend: str = "elo_poisson"  # one of: elo_poisson | lightgbm | xgboost
    target: str = "goals"  # goals | result
    params: dict[str, float | int | str] = Field(default_factory=dict)

    @field_validator("backend")
    @classmethod
    def _known_backend(cls, v: str) -> str:
        allowed = {"elo_poisson", "lightgbm", "xgboost", "catboost"}
        if v not in allowed:
            raise ValueError(f"backend must be one of {allowed}")
        return v


class AppConfig(BaseModel):
    """Root config object passed through the whole pipeline."""

    project_name: str = "wc2026-predictor"
    paths: PathsConfig = Field(default_factory=PathsConfig)
    elo: EloConfig = Field(default_factory=EloConfig)
    poisson: PoissonConfig = Field(default_factory=PoissonConfig)
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> AppConfig:
        """Load and validate config from a YAML file."""
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.model_validate(raw)
