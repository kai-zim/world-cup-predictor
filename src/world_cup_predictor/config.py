from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    random_seed: int = 42
    training_data_path: str = "data/matches.csv"
    output_dir: str = "outputs"
    mlflow_tracking_uri: str = "mlruns"
    n_simulations: int = 1000

    model_config = SettingsConfigDict(env_prefix="WCP_", env_file=".env", extra="ignore")
