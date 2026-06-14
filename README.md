# World Cup Predictor

Production-ready Python machine learning starter project for predicting FIFA World Cup knockout stage matches after the group stage.

## Highlights

- `src/` layout package: `world_cup_predictor`
- Poetry dependency management
- Feature modules for ingestion, engineering, Elo, xG, training, prediction and simulation
- Monte Carlo tournament simulation and Poisson match simulation
- MLflow experiment tracking hooks
- Streamlit dashboard starter app
- Pytest tests
- GitHub Actions CI
- Pydantic settings-based configuration and structured logging

## Project structure

- `src/`
- `data/`
- `notebooks/`
- `configs/`
- `models/`
- `outputs/`
- `tests/`
- `app/`

## Quickstart

```bash
poetry install
poetry run pytest
poetry run streamlit run app/streamlit_app.py
```

## Configuration

Environment-based settings are defined in `src/world_cup_predictor/config.py`.

## CI

GitHub Actions workflow: `.github/workflows/ci.yml`
