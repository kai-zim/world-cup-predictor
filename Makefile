.PHONY: install test lint format train calibrate predict simulate app download clean

## Install the package and all dev dependencies
install:
	pip install -e ".[dev]"

## Run the full test suite with coverage
test:
	pytest

## Run linters (ruff + mypy)
lint:
	ruff check src tests
	mypy src

## Auto-format code with ruff
format:
	ruff format src tests
	ruff check --fix src tests

## Download required data from Kaggle (needs kaggle CLI + API token, see .env.example)
download:
	python scripts/download_data.py

## Calibrate EloPoissonModel constants on real data (requires data/processed/results.csv)
calibrate:
	wc2026 calibrate \
		--history data/processed/results.csv \
		--out configs/calibrated_constants.yaml

## Run a forecast from real data files (requires data/processed/results.csv)
predict:
	wc2026 forecast \
		--history data/processed/results.csv \
		--seeding data/processed/wc2026_seeding_example.csv \
		--config configs/after_group_stage.yaml

## Train the learned goal model (requires data/processed/results.csv)
train:
	python -c "\
from wc2026.config.schema import AppConfig; \
from wc2026.models.train import train; \
m = train('data/processed/results.csv', AppConfig()); \
m.save('models/goal_model.joblib'); \
print('Model saved to models/goal_model.joblib')"

## Run the synthetic end-to-end Monte-Carlo demo (no data needed)
simulate:
	wc2026 demo --n 100000

## Launch the Streamlit dashboard
app:
	streamlit run app/dashboard.py

## Remove generated artefacts (does not touch data/ or models/)
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage