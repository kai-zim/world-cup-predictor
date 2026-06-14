"""Tests for configuration loading and validation (wc2026.config.schema)."""

from __future__ import annotations

from pathlib import Path

import pytest

from wc2026.config.schema import AppConfig, SimulationConfig


def test_default_config_loads_without_arguments() -> None:
    cfg = AppConfig()
    assert cfg.elo.start_rating == 1500.0
    assert cfg.simulation.n_simulations == 100_000
    assert cfg.simulation.random_seed == 42


def test_config_from_yaml_overrides_defaults(tmp_path: Path) -> None:
    yaml_content = "simulation:\n  n_simulations: 5000\n  random_seed: 7\n"
    p = tmp_path / "test.yaml"
    p.write_text(yaml_content)
    cfg = AppConfig.from_yaml(p)
    assert cfg.simulation.n_simulations == 5000
    assert cfg.simulation.random_seed == 7


def test_config_from_yaml_partial_override_keeps_defaults(tmp_path: Path) -> None:
    p = tmp_path / "partial.yaml"
    p.write_text("simulation:\n  n_simulations: 1000\n")
    cfg = AppConfig.from_yaml(p)
    assert cfg.simulation.n_simulations == 1000
    assert cfg.elo.start_rating == 1500.0  # unchanged


def test_invalid_backend_raises() -> None:
    with pytest.raises(Exception):
        AppConfig.model_validate({"model": {"backend": "not_a_real_backend"}})


def test_zero_simulations_raises() -> None:
    with pytest.raises(Exception):
        SimulationConfig(n_simulations=0)


def test_negative_simulations_raises() -> None:
    with pytest.raises(Exception):
        SimulationConfig(n_simulations=-100)


def test_world_cup_importance_exceeds_friendly() -> None:
    cfg = AppConfig()
    assert cfg.elo.importance["FIFA World Cup"] > cfg.elo.importance["Friendly"]


def test_all_known_backends_accepted() -> None:
    for backend in ("elo_poisson", "lightgbm", "xgboost", "catboost"):
        cfg = AppConfig.model_validate({"model": {"backend": backend}})
        assert cfg.model.backend == backend


def test_default_yaml_is_valid() -> None:
    """Regression: configs/default.yaml must always parse as a valid AppConfig."""
    yaml_path = Path("configs/default.yaml")
    if not yaml_path.exists():
        pytest.skip("configs/default.yaml not found (run from repo root)")
    cfg = AppConfig.from_yaml(yaml_path)
    assert cfg.project_name == "wc2026-predictor"
    assert cfg.elo.start_rating > 0


def test_per_round_configs_are_valid() -> None:
    """All per-round configs in configs/ must parse without error."""
    config_dir = Path("configs")
    if not config_dir.exists():
        pytest.skip("configs/ directory not found")
    yamls = list(config_dir.glob("after_*.yaml"))
    if not yamls:
        pytest.skip("No after_*.yaml configs found yet")
    for p in yamls:
        cfg = AppConfig.from_yaml(p)
        assert cfg.simulation.n_simulations > 0, f"{p.name} has invalid n_simulations"