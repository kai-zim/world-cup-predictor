from pathlib import Path

import pytest

from worldcup_predictor.data.loaders import load_raw_historical, load_raw_wc2026_from_dir
from worldcup_predictor.data.preprocessing import build_historical_match_table, build_match_table
from worldcup_predictor.features.feature_pipeline import build_feature_frame, build_historical_feature_frame
from worldcup_predictor.utils.config import FeaturesConfig, ModelConfig, SimulationConfig

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "wc2026"
HISTORICAL_FIXTURE = Path(__file__).parent / "fixtures" / "piterfm_sample.csv"


@pytest.fixture(scope="session")
def raw_wc2026():
    return load_raw_wc2026_from_dir(FIXTURES_DIR)


@pytest.fixture(scope="session")
def matches(raw_wc2026):
    return build_match_table(raw_wc2026)


@pytest.fixture(scope="session")
def features_config():
    return FeaturesConfig.load()


@pytest.fixture(scope="session")
def model_config():
    return ModelConfig.load()


@pytest.fixture(scope="session")
def simulation_config():
    return SimulationConfig.load()


@pytest.fixture(scope="session")
def feature_frame(raw_wc2026, features_config):
    return build_feature_frame(raw_wc2026, features_config)


@pytest.fixture(scope="session")
def played_matches(feature_frame):
    return feature_frame[feature_frame["winner"].notna()].copy()


@pytest.fixture(scope="session")
def raw_historical():
    return load_raw_historical(HISTORICAL_FIXTURE)


@pytest.fixture(scope="session")
def historical_matches(raw_historical):
    return build_historical_match_table(raw_historical)


@pytest.fixture(scope="session")
def historical_feature_frame(historical_matches, features_config):
    return build_historical_feature_frame(historical_matches, features_config)
