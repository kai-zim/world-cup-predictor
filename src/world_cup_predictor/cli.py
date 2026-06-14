from world_cup_predictor.config import Settings
from world_cup_predictor.logging_config import configure_logging


def main() -> None:
    configure_logging()
    _ = Settings()


if __name__ == "__main__":
    main()
